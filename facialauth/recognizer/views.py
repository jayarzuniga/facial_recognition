from django.shortcuts import render
from django.http import JsonResponse
from .models import UserFace
import numpy as np
import pickle
from django.views.decorators.http import require_POST
from django.core.exceptions import BadRequest
from django.views.decorators.csrf import csrf_exempt
import cv2
import logging
from deepface import DeepFace
import os

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_CONTENT_TYPES = ['image/jpeg', 'image/png']
FACE_DB_PATH = 'face_db/' 
os.makedirs(FACE_DB_PATH, exist_ok=True)

def validate_image_file(file):
    if file.size > MAX_IMAGE_SIZE:
        raise BadRequest("Image size too large (max 5MB)")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise BadRequest("Unsupported image format (only JPEG/PNG allowed)")

def save_temp_image(file, filename):
    try:
        validate_image_file(file)
        filepath = os.path.join(FACE_DB_PATH, filename)
        with open(filepath, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)
        return filepath
    except Exception as e:
        logger.error(f"Image save failed: {str(e)}")
        raise BadRequest("Failed to process image")

def index(request):
    return render(request, 'recognizer/index.html')

@csrf_exempt
@require_POST
def register_face(request):
    try:
        logger.info("Register face request received")
        name = request.POST.get('name')
        image_file = request.FILES.get('image')

        if not name or not name.strip():
            return JsonResponse({'success': False, 'message': 'Name is required'})
        if not image_file:
            return JsonResponse({'success': False, 'message': 'Image is required'})

        # Save image
        filename = f"{name.strip().replace(' ', '_')}.jpg"
        filepath = save_temp_image(image_file, filename)

        return JsonResponse({'success': True, 'message': f'Face registered for {name}'})

    except BadRequest as e:
        logger.warning(f"Bad request: {str(e)}")
        return JsonResponse({'success': False, 'message': str(e)}, status=400)
    except Exception as e:
        logger.error(f"Error in register_face: {str(e)}", exc_info=True)
        return JsonResponse({'success': False, 'message': 'Server error'}, status=500)

@csrf_exempt
@require_POST
def authenticate_face(request):
    try:
        logger.info("Authenticating face")
        image_file = request.FILES.get('image')

        if not image_file:
            return JsonResponse({'success': False, 'message': 'Image is required'})

        temp_path = save_temp_image(image_file, 'temp_auth.jpg')

        results = DeepFace.find(img_path=temp_path, db_path=FACE_DB_PATH, enforce_detection=True)
        if results and len(results[0]) > 0:
            identity = os.path.basename(results[0].iloc[0]['identity'])
            name = os.path.splitext(identity)[0].replace('_', ' ')
            return JsonResponse({'success': True, 'message': f'Authenticated as {name}', 'name': name})
        else:
            return JsonResponse({'success': False, 'message': 'Face not recognized'})

    except BadRequest as e:
        logger.warning(f"Bad request in auth: {str(e)}")
        return JsonResponse({'success': False, 'message': str(e)}, status=400)
    except Exception as e:
        logger.error(f"Error in authenticate_face: {str(e)}", exc_info=True)
        return JsonResponse({'success': False, 'message': 'Server error'}, status=500)
