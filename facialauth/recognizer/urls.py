from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register_face, name='register_face'),
    path('authenticate/', views.authenticate_face, name='authenticate_face'),
]