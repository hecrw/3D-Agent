"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from chat_interface import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('chat/<int:session_id>/', views.chat_detail, name='chat_detail'),
    path('chat/new/', views.new_chat, name='new_chat'),
    path('chat/<int:session_id>/delete/', views.delete_chat, name='delete_chat'),
    path('chat/<int:session_id>/rename/', views.rename_chat, name='rename_chat'),
    path('api/chat/<int:session_id>/message/', views.api_send_message, name='api_send_message'),
    
    path('gallery/', views.gallery, name='gallery'),
    path('api/gallery/delete/', views.api_delete_assets, name='api_delete_assets'),
]

# This tells Django: "If someone asks for /media/..., look in the MEDIA_ROOT folder"
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)