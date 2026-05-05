from django.db import models
from django.utils.timezone import now

class ChatSession(models.Model):
    title = models.CharField(max_length=100, default="New Chat")
    created_at = models.DateTimeField(default=now)

    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['-created_at'] # Newest chats at the top

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.SET_NULL, null=True, related_name="messages")
    sender = models.CharField(max_length=50) # 'user' or 'assistant'
    text = models.TextField()
    object_path = models.URLField(blank=True, null=True) # For the 3D GLB file
    
    # NEW: Stores the Modal function call ID (e.g., fc-xxxxxx) 
    # to allow the Stop button to cancel the specific GPU task.
    modal_call_id = models.CharField(max_length=255, blank=True, null=True)
    attachment = models.ImageField(upload_to='chat_attachments/', blank=True, null=True)
    created_at = models.DateTimeField(default=now)

    class Meta:
        ordering = ['created_at'] # Oldest to newest for chat history