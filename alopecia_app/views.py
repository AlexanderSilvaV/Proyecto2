from django.shortcuts import render
from .forms import ChatForm
from ollama_client import send_prompt
from .cnn_model import predict_image, load_model
import os
from django.conf import settings
import pandas as pd
import torch

# Configuración de rutas y modelos
model_path = os.path.join(settings.BASE_DIR, 'alopecia_cnn.pth')
csv_file = os.path.join(settings.BASE_DIR, 'dataset/bald_people/bald_people.csv')

AVAILABLE_MODELS_LLM = [
    "qwen2.5:7b",
    "dolphin-phi:2.7b", 
    "llama3:8b",
    "mistral:latest"
]

AVAILABLE_MODELS_IMAGE = [
    'mobilenetv2',
    'efficientnet_b0',
    'custom_cnn'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar anotaciones y modelo
try:
    annotations_df = pd.read_csv(csv_file)
    class_names = sorted(annotations_df['type'].unique())
    num_classes = len(class_names)
    model_type = "efficientnet_b0"
    model = load_model(model_path, model_type, num_classes, device)
    print(f"✅ Modelo {model_type} cargado exitosamente en {device}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None
    class_names = []

def home(request):
    return render(request, 'alopecia_app/home.html')

def chat_with_ollama(request):
    response = None
    prediction = None
    image_url = None
    error_message = None

    if request.method == 'POST':
        form = ChatForm(request.POST, request.FILES)
        selected_llm_model = request.POST.get('llm_model')
        selected_image_model = request.POST.get('image_model')

        if form.is_valid():
            prompt = form.cleaned_data.get('prompt')
            uploaded_image = form.cleaned_data.get('image')

            # Procesar imagen si se subió
            if uploaded_image:
                try:
                    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                    image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)

                    with open(image_path, 'wb+') as f:
                        for chunk in uploaded_image.chunks():
                            f.write(chunk)

                    image_url = os.path.join(settings.MEDIA_URL, uploaded_image.name)

                    # Buscar etiqueta real
                    image_name = uploaded_image.name
                    match = annotations_df[annotations_df['images'].str.contains(image_name)]
                    real_label = match.iloc[0]['type'] if not match.empty else "Desconocida"

                    if model is not None:
                        pred_label = predict_image(image_path, model, class_names, device)
                        print(f"✅ Etiqueta real: {real_label}")
                        print(f"🔮 Etiqueta predicha: {pred_label}")
                        prediction = f"Real: {real_label} | Predicha: {pred_label}"
                    else:
                        prediction = "Error: Modelo de imagen no disponible"

                except Exception as e:
                    error_message = f"Error procesando imagen: {str(e)}"
                    print(f"❌ {error_message}")

            # Procesar texto si hay prompt
            if prompt:
                if selected_llm_model and selected_llm_model in AVAILABLE_MODELS_LLM:
                    try:
                        response = send_prompt(prompt, model_name=selected_llm_model)
                        print(f"🤖 Respuesta LLM ({selected_llm_model}): {response[:100]}...")
                    except Exception as e:
                        error_message = f"Error con modelo LLM: {str(e)}"
                        print(f"❌ {error_message}")
                else:
                    error_message = "Por favor selecciona un modelo LLM válido"

            # Validación adicional
            if uploaded_image and selected_image_model not in AVAILABLE_MODELS_IMAGE:
                if not error_message:
                    error_message = "Por favor selecciona un modelo de imagen válido"

        else:
            error_message = "Formulario inválido. Por favor revisa los campos."
    else:
        form = ChatForm()

    context = {
        'form': form,
        'response': response,
        'prediction': prediction,
        'image_url': image_url,
        'error_message': error_message,
        'available_models_llm': AVAILABLE_MODELS_LLM,
        'available_models_image': AVAILABLE_MODELS_IMAGE,
        'device_info': str(device),
        'model_loaded': model is not None,
        'num_classes': len(class_names),
    }

    return render(request, 'alopecia_app/chat.html', context)
