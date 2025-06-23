# alopecia_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),                           # Página principal: /
    path('chat/', views.chat_with_ollama, name='chat_with_ollama'),  # Chat: /chat/
]