from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .models import ChatSession, ChatMessage  # Confirmed model name
import json
import re
from agent import process_chat

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
    # 1. Start with a default title
    initial_prompt = ""
    title = "New Chat"

    if request.method == "POST":
        # Logic for when someone uses the big central input on the landing page
        initial_prompt = request.POST.get("initial_prompt", "").strip()
        if initial_prompt:
            title = initial_prompt[:30]

    # 2. Create the session (happens for both GET and POST)
    session = ChatSession.objects.create(title=title)

    # 3. If it was a POST with a prompt, save the message
    if request.method == "POST" and initial_prompt:
        ChatMessage.objects.create(
            session=session,
            sender="user",
            text=initial_prompt
        )
    
    # 4. ALWAYS return a redirect here (outside the 'if' block)
    # This fixes the "Returned None" error for GET requests.
    return redirect("chat_detail", session_id=session.id)

def api_send_message(request, session_id):
    """This is what the JavaScript calls to get the AI's 3D response."""
    if request.method == "POST":
        data = json.loads(request.body)
        user_text = data.get('text')
        active_session = get_object_or_404(ChatSession, id=session_id)

        # --- THE CHECK ---
        # Get the absolute last message in this session
        # Inside api_send_message...
        last_msg = active_session.messages.all().order_by('created_at').last()
        if not (last_msg and last_msg.sender == "user" and last_msg.text == user_text):
            ChatMessage.objects.create(session=active_session, sender="user", text=user_text)

        # Build history for the agent
        db_messages = active_session.messages.all().order_by('created_at')
        history = [{"role": msg.sender, "content": msg.text} for msg in db_messages]

        # Call your 3D Agent
        output = process_chat(user_text, history)
        
        # Extract response text
        raw_text = output[-1].get('text', str(output)) if isinstance(output, list) else str(output)
        
        # Path Extraction for 3D assets
        bot_3d_path = ""
        file_match = re.search(r'3d_outputs[/\\](.+?\.(?:glb|png))', raw_text)
        if file_match:
            bot_3d_path = f"/media/3d_outputs/{file_match.group(1)}"

        # Clean and supplement bot text
        bot_text = scrub_bot_text(raw_text)
        if bot_3d_path and "preview window" not in bot_text:
            bot_text += "\n\nYou can view it in the preview window."

        # Save Assistant message to DB
        ChatMessage.objects.create(
            session=active_session, 
            sender="assistant", 
            text=bot_text, 
            object_path=bot_3d_path
        )

        return JsonResponse({"text": bot_text, "3d_object_path": bot_3d_path})

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


def gallery(request):
    all_sessions = ChatSession.objects.all()
    generated_assets = ChatMessage.objects.exclude(object_path__isnull=True).exclude(object_path__exact='').order_by('-created_at')
    
    return render(request, 'chat_interface/gallery.html', {
        'all_sessions': all_sessions,
        'assets': generated_assets
    })