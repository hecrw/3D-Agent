from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import ChatSession, ChatMessage
import json
import os
import re
from urllib.parse import urlparse, unquote
from agent import generate_chat_title, process_chat_stream
from .runs import start_run, get_run
# NEW: Import Modal to handle cancellation
import modal


def _local_path_for_object(object_path):
    """Resolve a stored object_path (a /media/... URL) to an absolute local file
    path on disk, or None if it is an external URL or no longer present. Used to
    hand the agent the file of a previously generated asset so the user can refer
    back to it ("texture the last one")."""
    if not object_path:
        return None
    parsed = urlparse(object_path)
    if parsed.scheme in ("http", "https"):
        return None
    path = unquote(parsed.path or object_path)
    media_url = settings.MEDIA_URL.rstrip("/") if settings.MEDIA_URL else ""
    if media_url and path.startswith(media_url + "/"):
        full = os.path.join(settings.MEDIA_ROOT, path[len(media_url) + 1:])
    elif os.path.isabs(path):
        full = path
    else:
        full = os.path.join(settings.MEDIA_ROOT, path)
    return full if os.path.isfile(full) else None


def _delete_local_asset(object_path):
    """Remove the file under MEDIA_ROOT that object_path points to. Skips external URLs."""
    if not object_path:
        return
    parsed = urlparse(object_path)
    if parsed.scheme in ("http", "https"):
        return  # external image, nothing local to remove
    path = unquote(parsed.path or object_path)
    media_url = settings.MEDIA_URL.rstrip("/") if settings.MEDIA_URL else ""
    if media_url and path.startswith(media_url + "/"):
        rel = path[len(media_url) + 1:]
        full = os.path.join(settings.MEDIA_ROOT, rel)
    elif os.path.isabs(path):
        full = path
    else:
        full = os.path.join(settings.MEDIA_ROOT, path)
    try:
        if os.path.isfile(full):
            os.remove(full)
    except OSError:
        pass

# --- HELPER: CLEAN THE BOT OUTPUT ---
def scrub_bot_text(raw_text):
    # 1. Try to handle JSON-like structures from the model
    try:
        # Some models might return valid JSON
        if raw_text.strip().startswith('{') and raw_text.strip().endswith('}'):
            data = json.loads(raw_text)
            if isinstance(data, dict):
                if "content" in data: raw_text = data["content"]
                elif "text" in data: raw_text = data["text"]
                elif "action" in data and isinstance(data["action"], str) and len(data) == 1:
                    pass # Just the action, keep it
    except:
        # If not valid JSON, try a regex to pull text out of a "content": "..." pattern
        json_text_match = re.search(r'"(?:content|text)"\s*:\s*"(.+?)"', raw_text, re.DOTALL)
        if json_text_match:
            raw_text = json_text_match.group(1).encode().decode('unicode_escape')

    # 2. Clean up paths and internal info
    # Strip internal context tags the model may have echoed back into its reply
    # (e.g. "[Previously generated asset: /…]", "[Uploaded Image Local Path: /…]"),
    # including any surrounding markdown asterisks and a possibly-truncated bracket.
    raw_text = re.sub(
        r'\*{0,2}\[(?:Previously generated asset|Uploaded Image Local Path):[^\]]*\]?\*{0,2}',
        '', raw_text)
    # We NO LONGER remove markdown links [text](url)
    clean_text = re.sub(r'[^\s]*3d_outputs[^\s]*', '', raw_text)
    clean_text = re.sub(r'[^.!?\n]*(?:/Users/|/home/|/var/|/media/|[a-zA-Z]:\\)[^.!?\n]*[.!?]?', '', clean_text)
    
    # 3. Final polish
    clean_text = clean_text.strip()
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
    
    return clean_text if clean_text else "I've generated the results for you."

# --- VIEWS ---

def index(request):
    """The landing page with the big central input."""
    all_sessions = ChatSession.objects.all().order_by('-created_at')
    return render(request, 'chat_interface/index.html', {
        'all_sessions': all_sessions,
        'active_session': None,
        'messages': []
    })

def chat_detail(request, session_id):
    """The actual chat interface."""
    active_session = get_object_or_404(ChatSession, id=session_id)
    all_sessions = ChatSession.objects.all().order_by("-created_at") # Fixed order_by
    messages = active_session.messages.all().order_by("created_at")
    
    return render(request, "chat_interface/index.html", { # Ensure path is correct
        "active_session": active_session,
        "all_sessions": all_sessions,
        "messages": messages
    })

def new_chat(request):
    """Creates a session and redirects instantly."""
    # Delete any existing sessions that have no messages to keep the history clean
    from django.db.models import Count
    ChatSession.objects.annotate(msg_count=Count('messages')).filter(msg_count=0).delete()

    # 1. Start with a default title
    initial_prompt = ""
    title = "New Chat"

    if request.method == "POST":
        # Logic for when someone uses the big central input on the landing page
        initial_prompt = request.POST.get("initial_prompt", "").strip()
        if initial_prompt:
            title = initial_prompt[:30]

    # 2. Create the session
    session = ChatSession.objects.create(title=title)

    # 3. If it was a POST with a prompt, save message and generate snappy title
    if request.method == "POST" and initial_prompt:
        # Generate snappy title
        session.title = generate_chat_title(initial_prompt)
        session.save()

        ChatMessage.objects.create(
            session=session,
            sender="user",
            text=initial_prompt,
            attachment=request.FILES.get('attachment')
        )
    
    return redirect(f"/chat/{session.id}/?auto_start=true")


def delete_chat(request, session_id):
    chat = get_object_or_404(ChatSession, id=session_id)
    messages = list(chat.messages.all())
    for msg in messages:
        if msg.attachment:
            try:
                msg.attachment.delete(save=False)
            except Exception:
                pass
        _delete_local_asset(msg.object_path)
    # FK is SET_NULL, so delete messages explicitly to avoid orphans in the gallery
    chat.messages.all().delete()
    chat.delete()
    return redirect('index')


def rename_chat(request, session_id):
    if request.method == "POST":
        chat = get_object_or_404(ChatSession, id=session_id)
        new_title = request.POST.get('new_title', 'Unnamed Chat')
        chat.title = new_title
        chat.save()
    return redirect('chat_detail', session_id=session_id)



def api_send_message(request, session_id):
    """JS calls this to get the AI's response (streamed via SSE)."""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=400)
    
    # Two transports: JSON (text + base64 image, the default) or multipart
    # (text + an uploaded 3D mesh file, which is too big to base64 in JSON).
    mesh_file = None
    if request.content_type and request.content_type.startswith("multipart"):
        user_text = request.POST.get("text", "")
        user_image_b64 = None
        mesh_file = request.FILES.get("mesh")
    else:
        data = json.loads(request.body)
        user_text = data.get('text')
        user_image_b64 = data.get('image')  # base64 string
    active_session = get_object_or_404(ChatSession, id=session_id)

    last_msg = active_session.messages.all().order_by('created_at').last()
    current_user_msg = None
    if not (last_msg and last_msg.sender == "user" and last_msg.text == user_text):
        import base64
        from django.core.files.base import ContentFile

        attachment = None
        if user_image_b64 and "," in user_image_b64:
            format, imgstr = user_image_b64.split(';base64,')
            ext = format.split('/')[-1]
            attachment = ContentFile(base64.b64decode(imgstr), name=f"upload.{ext}")

        current_user_msg = ChatMessage.objects.create(
            session=active_session,
            sender="user",
            text=user_text,
            attachment=attachment,
            mesh_upload=mesh_file,
        )

    # Fetch history EXCLUDING the message we are currently processing (the last one)
    db_messages = active_session.messages.all().order_by('created_at')
    # If the last message is the one we just processed/identified, remove it from history
    # to avoid the AI seeing the same prompt twice.
    history_messages = db_messages[:db_messages.count()-1] if db_messages.count() > 0 else []
    
    history = []
    for msg in history_messages:
        content = msg.text
        if msg.attachment:
            content += f"\n\n[Uploaded Image Local Path: {msg.attachment.path}]"
        if msg.mesh_upload:
            content += f"\n\n[Uploaded 3D Asset Local Path: {msg.mesh_upload.path}]"
        # Let the agent reference assets it produced earlier ("the last one").
        if msg.sender == "assistant" and msg.object_path:
            asset_path = _local_path_for_object(msg.object_path)
            if asset_path:
                content += f"\n\n[Previously generated asset: {asset_path}]"

        item = {"role": msg.sender, "content": content}
        if msg.attachment:
            # Provide absolute URL for the agent (LangChain/Gemini)
            item["image"] = request.build_absolute_uri(msg.attachment.url)
        history.append(item)

    # The current image for the agent (if just sent)
    current_image_url = None
    if current_user_msg and current_user_msg.attachment:
        current_image_url = request.build_absolute_uri(current_user_msg.attachment.url)
        # We append the local path to the prompt so the agent knows what path to pass to tools
        user_text += f"\n\n[Uploaded Image Local Path: {current_user_msg.attachment.path}]"

    # Hand the agent the absolute path of an uploaded 3D mesh so it can feed it
    # straight to the mesh tools (compose_scene, render, texture).
    if current_user_msg and current_user_msg.mesh_upload:
        user_text += f"\n\n[Uploaded 3D Asset Local Path: {current_user_msg.mesh_upload.path}]"

    if active_session.title == "New Chat" or active_session.title == user_text[:30]:
        active_session.title = generate_chat_title(user_text)
        active_session.save()

    # Run the generation in a background thread (owned by start_run) so it
    # survives a browser refresh: the worker keeps going and persists the reply
    # even if this HTTP response disconnects. We subscribe this request to the
    # run's event stream. If a run is already active for this session, start_run
    # returns it instead of starting a second one (dedupes double submits).
    handle = start_run(
        session_id,
        lambda: _stream_agent_events(
            process_chat_stream(user_text, history,
                                user_image_url=current_image_url,
                                session_id=session_id),
            active_session,
        ),
    )
    return StreamingHttpResponse(handle.subscribe(), content_type='text/event-stream')


def _media_url_for_local_path(path):
    """Map an absolute MEDIA_ROOT path (e.g. a restyled image the agent is about
    to send to a pipeline) to its browser /media/ URL, so the approval gate can
    show the image. Returns "" if the path isn't under MEDIA_ROOT."""
    if not path:
        return ""
    try:
        rel = os.path.relpath(os.path.abspath(path), settings.MEDIA_ROOT)
    except ValueError:
        return ""
    if rel.startswith(".."):
        return ""
    media_url = settings.MEDIA_URL.rstrip("/") if settings.MEDIA_URL else "/media"
    return f"{media_url}/{rel.replace(os.sep, '/')}"


def _stream_agent_events(events, active_session):
    """Translate the agent's event dicts into SSE frames, persisting the final
    assistant message. Shared by the initial send and the resume endpoint."""
    for event in events:
        if event["type"] == "call_id":
            yield f"data: {json.dumps({'type': 'call_id', 'modal_call_id': event['content']})}\n\n"

        elif event["type"] == "status":
            yield f"data: {json.dumps(event)}\n\n"

        elif event["type"] == "interrupt":
            # Agent paused for human approval before a 3D-generation tool. Do NOT
            # save an assistant message — the run is suspended in the checkpointer
            # awaiting a decision posted to the resume endpoint.
            payload = {
                "type": "interrupt",
                "tool": event.get("tool", ""),
                "label": event.get("label", ""),
                "image_url": _media_url_for_local_path(event.get("image_path", "")),
                "args": event.get("args", {}),
                "allowed_decisions": event.get("allowed_decisions", []),
            }
            yield f"data: {json.dumps(payload)}\n\n"

        elif event["type"] == "text":
            raw_text = event["content"]   # plain string now
            print(f"RAW BOT TEXT: {raw_text}")

            bot_3d_path = ""
            # Prefer the mesh path harvested from tool output (the reply text no
            # longer contains paths); fall back to scraping the text.
            path_source = event.get("artifact") or raw_text
            file_match = re.search(r'3d_outputs[/\\](.+?\.(?:glb|png|jpe?g|webp|gif))', path_source, re.IGNORECASE)
            if file_match:
                filename = file_match.group(1).lstrip('/\\')
                bot_3d_path = f"/media/3d_outputs/{filename}"
            else:
                # Check for external URLs if no local output found (e.g. from image search)
                url_match = re.search(r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp))', raw_text, re.IGNORECASE)
                if url_match:
                    bot_3d_path = url_match.group(1)

            bot_text = scrub_bot_text(raw_text)
            if bot_3d_path and "3d_outputs" in path_source and "preview window" not in bot_text:
                bot_text += "\n\nYou can view it in the chat."

            ChatMessage.objects.create(
                session=active_session,
                sender="assistant",
                text=bot_text,
                object_path=bot_3d_path,
            )

            final_data = {
                "type": "final",
                "text": bot_text,
                "3d_object_path": bot_3d_path,
            }
            yield f"data: {json.dumps(final_data)}\n\n"


def api_resume_chat(request, session_id):
    """Resume a run paused at the human approval gate.

    Body: {"decisions": [{"type": "approve"} | {"type": "reject", "message": ...}
                          | {"type": "edit", "edited_action": {...}}]}
    Streams the continued agent run via SSE, exactly like api_send_message.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=400)

    from agent import resume_chat_stream
    active_session = get_object_or_404(ChatSession, id=session_id)
    data = json.loads(request.body)
    decisions = data.get("decisions") or []

    handle = start_run(
        session_id,
        lambda: _stream_agent_events(
            resume_chat_stream(session_id, decisions),
            active_session,
        ),
    )
    return StreamingHttpResponse(handle.subscribe(), content_type='text/event-stream')


def api_reconnect(request, session_id):
    """Re-attach to a generation already in flight for this session, so a browser
    refresh does not lose (or resubmit) it. Three cases:

      1. A run is still active  -> stream it (replays everything, then tails).
      2. The run paused at the approval gate before we disconnected -> re-emit
         the gate so the user can still approve/reject.
      3. The run finished while we were away -> emit its saved final message.

    Otherwise returns {"active": false} and the client does nothing.
    """
    active_session = get_object_or_404(ChatSession, id=session_id)

    handle = get_run(session_id)
    if handle is not None:
        print(f"[reconnect {session_id}] -> live run (streaming handle)")
        return StreamingHttpResponse(handle.subscribe(), content_type='text/event-stream')

    from agent import peek_interrupt
    gate = peek_interrupt(session_id)
    if gate is not None:
        print(f"[reconnect {session_id}] -> re-emitting approval gate")
        def gate_frame():
            payload = {
                "type": "interrupt",
                "tool": gate.get("tool", ""),
                "label": gate.get("label", ""),
                "image_url": _media_url_for_local_path(gate.get("image_path", "")),
                "args": gate.get("args", {}),
                "allowed_decisions": gate.get("allowed_decisions", []),
            }
            yield f"data: {json.dumps(payload)}\n\n"
        return StreamingHttpResponse(gate_frame(), content_type='text/event-stream')

    last = active_session.messages.order_by("created_at").last()
    if last is not None and last.sender == "assistant":
        print(f"[reconnect {session_id}] -> replaying saved final message")
        def final_frame():
            payload = {
                "type": "final",
                "text": last.text,
                "3d_object_path": last.object_path or "",
            }
            yield f"data: {json.dumps(payload)}\n\n"
        return StreamingHttpResponse(final_frame(), content_type='text/event-stream')

    # No live run, no gate, last message is the user's prompt: a generation was
    # in flight but its background thread is gone (almost always the dev server
    # auto-reloaded on a file save, which kills daemon threads + the in-memory
    # registry). Surface it instead of vanishing silently.
    from agent import peek_interrupt  # noqa: F811  (already imported above)
    state_next = None
    try:
        import agent as _agent
        state_next = _agent.agent.get_state(_agent._thread_config(session_id)).next
    except Exception:
        pass
    print(f"[reconnect {session_id}] -> no live run, no gate (orphaned? next={state_next})")
    if state_next:  # a checkpoint exists mid-run -> it was interrupted
        def orphan_frame():
            yield f"data: {json.dumps({'type': 'status', 'content': 'The previous generation was interrupted — please resend.'})}\n\n"
            yield f"data: {json.dumps({'type': 'final', 'text': 'That generation was interrupted (the server restarted). Please send the request again.', '3d_object_path': ''})}\n\n"
        return StreamingHttpResponse(orphan_frame(), content_type='text/event-stream')

    return JsonResponse({"active": False})


def api_stop_chat(request):
    """Kills the running Modal container using the call_id."""
    if request.method == "POST":
        # Check if it's form-encoded (from my previous JS update) or JSON
        call_id = request.POST.get('call_id')
        if not call_id:
            try:
                data = json.loads(request.body)
                call_id = data.get('call_id')
            except: pass

        if call_id:
            try:
                # Interact with Modal API to terminate the specific function call
                f_call = modal.functions.FunctionCall.from_id(call_id)
                f_call.cancel()
                return JsonResponse({"status": "stopped", "call_id": call_id})
            except Exception as e:
                return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"error": "No call_id provided"}, status=400)


@csrf_exempt # Or ensure CSRF is handled in fetch
def api_delete_assets(request):
    if request.method == "POST":
        data = json.loads(request.body)
        ids = data.get('ids', [])
        ChatMessage.objects.filter(id__in=ids).delete()
        return JsonResponse({"status": "success"})
    return JsonResponse({"error": "Invalid method"}, status=400)


def gallery(request):
    all_sessions = ChatSession.objects.all()
    generated_assets = ChatMessage.objects.exclude(object_path__isnull=True).exclude(object_path__exact='').order_by('-created_at')
    
    return render(request, 'chat_interface/gallery.html', {
        'all_sessions': all_sessions,
        'assets': generated_assets
    })