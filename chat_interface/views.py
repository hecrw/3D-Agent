from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatSession, ChatMessage
import json
import re
from agent import process_chat, generate_chat_title

# --- HELPER: CLEAN THE BOT OUTPUT ---
def scrub_bot_text(raw_text):
    clean_text = re.sub(r'\[.*?\]\(.*?\)', '', raw_text) 
    clean_text = re.sub(r'[^\s]*3d_outputs[^\s]*', '', clean_text)
    clean_text = re.sub(r'[^.!?\n]*(?:/Users/|/home/|/var/|/media/|[a-zA-Z]:\\)[^.!?\n]*[.!?]?', '', clean_text)
    clean_text = clean_text.strip()
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
    return clean_text if clean_text else "I've generated the 3D model for you."

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
    
    return redirect("chat_detail", session_id=session.id)

from django.http import StreamingHttpResponse

def api_send_message(request, session_id):
    """This is what the JavaScript calls to get the AI's 3D response."""
    if request.method == "POST":
        data = json.loads(request.body)
        user_text = data.get('text')
        active_session = get_object_or_404(ChatSession, id=session_id)

        # Ensure user message is saved
        last_msg = active_session.messages.all().order_by('created_at').last()
        if not (last_msg and last_msg.sender == "user" and last_msg.text == user_text):
            ChatMessage.objects.create(session=active_session, sender="user", text=user_text)

        # Build history for the agent
        db_messages = active_session.messages.all().order_by('created_at')
        history = [{"role": msg.sender, "content": msg.text} for msg in db_messages]

        # Auto-title if this is the first real message in a generic chat
        if active_session.title == "New Chat" or active_session.title == user_text[:30]:
             active_session.title = generate_chat_title(user_text)
             active_session.save()

        from agent import process_chat_stream

        def event_stream():
            bot_text = ""
            bot_3d_path = ""
            
            for event in process_chat_stream(user_text, history):
                if event["type"] == "status":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "text":
                    raw_text = event["content"]
                    
                    # Path Extraction for 3D assets
                    file_match = re.search(r'3d_outputs[/\\](.+?\.(?:glb|png))', raw_text)
                    if file_match:
                        bot_3d_path = f"/media/3d_outputs/{file_match.group(1)}"

                    # Clean and supplement bot text
                    bot_text = scrub_bot_text(raw_text)
                    if bot_3d_path and "preview window" not in bot_text:
                        bot_text += "\n\nYou can view it in the chat."

                    # Save Assistant message to DB
                    ChatMessage.objects.create(
                        session=active_session, 
                        sender="assistant", 
                        text=bot_text, 
                        object_path=bot_3d_path
                    )
                    
                    final_data = {
                        "type": "final",
                        "text": bot_text,
                        "3d_object_path": bot_3d_path
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"

        return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

    return JsonResponse({"error": "Invalid method"}, status=400)

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