from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatSession, ChatMessage
import json
import re
from agent import generate_chat_title, process_chat_stream
# NEW: Import Modal to handle cancellation
import modal

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
            text=initial_prompt
        )
    
    return redirect(f"/chat/{session.id}/?auto_start=true")


def delete_chat(request, session_id):
    chat = get_object_or_404(ChatSession, id=session_id)
    chat.delete()
    # Always redirect to index (landing page) after deletion
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
    
    data = json.loads(request.body)
    user_text = data.get('text')
    active_session = get_object_or_404(ChatSession, id=session_id)

    last_msg = active_session.messages.all().order_by('created_at').last()
    if not (last_msg and last_msg.sender == "user" and last_msg.text == user_text):
        ChatMessage.objects.create(session=active_session, sender="user", text=user_text)

    db_messages = active_session.messages.all().order_by('created_at')
    history = [{"role": msg.sender, "content": msg.text} for msg in db_messages]

    if active_session.title == "New Chat" or active_session.title == user_text[:30]:
        active_session.title = generate_chat_title(user_text)
        active_session.save()

    def event_stream():
        for event in process_chat_stream(user_text, history):
            if event["type"] == "call_id":
                yield f"data: {json.dumps({'type': 'call_id', 'modal_call_id': event['content']})}\n\n"
            
            elif event["type"] == "status":
                yield f"data: {json.dumps(event)}\n\n"
            
            elif event["type"] == "text":
                raw_text = event["content"]   # plain string now
                print(f"RAW BOT TEXT: {raw_text}")

                bot_3d_path = ""
                # Check for local outputs first
                file_match = re.search(r'3d_outputs[/\\](.+?\.(?:glb|png))', raw_text)
                if file_match:
                    filename = file_match.group(1).lstrip('/\\')
                    bot_3d_path = f"/media/3d_outputs/{filename}"
                else:
                    # Check for external URLs if no local output found (e.g. from image search)
                    url_match = re.search(r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp))', raw_text, re.IGNORECASE)
                    if url_match:
                        bot_3d_path = url_match.group(1)

                bot_text = scrub_bot_text(raw_text)
                if bot_3d_path and "3d_outputs" in raw_text and "preview window" not in bot_text:
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

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

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