# 🧠 Alopecia Project

Este proyecto es un sistema de inteligencia artificial basado en visión computacional para la detección y clasificación automática de distintos tipos de alopecia a partir de imágenes del cuero cabelludo, utilizando la escala de Hamilton como referencia. La aplicación incluye un modelo de deep learning entrenado y una interfaz web para facilitar la interacción con el usuario.

---

## 🔍 Objetivo

Desarrollar una herramienta capaz de identificar automáticamente el tipo de alopecia (clasificación de Type 1 a Type 7) con el fin de apoyar diagnósticos preliminares en dermatología, especialmente en contextos con acceso limitado a especialistas.

---

## 🛠️ Tecnologías y herramientas utilizadas

- **Python 3.11**  
- **Django 4.x** (framework web para el desarrollo de la interfaz y backend)  
- **Laragon 5.0** (entorno de desarrollo local, para gestión de servidor y bases de datos)  
- **PyTorch** (framework de deep learning para entrenamiento y evaluación del modelo CNN personalizado)  
- **Git LFS** (para manejo eficiente de archivos de modelos grandes, tipo `.pth`)  
- Librerías auxiliares: numpy, pandas, torchvision, matplotlib, scikit-learn, Pillow

---

## 📁 Estructura del proyecto

alopecia_project/
├── alopecia_app/ # Aplicación Django: vistas, modelos, URLs, formularios

├── media/ # Carpeta para imágenes subidas por usuarios

├── models/ # Modelos entrenados (.pth)

├── static/ # Archivos estáticos: CSS, JavaScript, imágenes

├── templates/ # Archivos HTML para la interfaz web

├── cnn_model.py # Script para entrenamiento y evaluación del modelo CNN

├── manage.py # Script para gestión de comandos Django

├── requirements.txt # Dependencias del proyecto

├── .gitignore # Archivos y carpetas ignoradas por git

└── README.md # Documentación del proyecto

yaml
Copiar
Editar

---

## 🚀 Instrucciones para instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/AlexanderSilvaV/Proyecto2.git
cd Proyecto2
2. Configurar entorno virtual y dependencias
En Linux/macOS:

bash
Copiar
Editar
python3.11 -m venv venv
source venv/bin/activate
En Windows (PowerShell):

powershell
Copiar
Editar
python -m venv venv
.\venv\Scripts\activate
Instalar las librerías necesarias:

bash
Copiar
Editar
pip install -r requirements.txt
3. Configurar Laragon (opcional, para entorno local)
Asegúrate de que Laragon esté instalado (versión 5.0).

Configura Apache/Nginx para servir el proyecto Django (opcional).

Puedes usar Laragon para gestionar bases de datos si tu proyecto las requiere.

4. Ejecutar servidor de desarrollo Django
bash
Copiar
Editar
python manage.py runserver
5. Acceder a la aplicación
Abre tu navegador y visita:

cpp
Copiar
Editar
http://127.0.0.1:8000/
🧠 Detalles del modelo
Arquitectura: CNN personalizada entrenada desde cero adaptada para clasificación en 7 clases.

Función de pérdida: CrossEntropy y Focal Loss para mitigar desequilibrio en clases.

Entrenamiento: Usando imágenes de cuero cabelludo clasificadas según la escala de Hamilton.

Resultados: Métricas reportadas incluyen accuracy, matriz de confusión, curvas de precisión y recall.

🧪 Evaluación
Monitoreo de accuracy y pérdida por época durante el entrenamiento.

Análisis de matriz de confusión para identificar posibles errores de clasificación entre tipos.

Evaluación con métricas de precisión y recall para cada clase, asegurando balance entre falsos positivos y negativos.

⚙️ Notas técnicas importantes
Se utiliza Git LFS para manejar los archivos .pth de modelos grandes (evita problemas con límite de tamaño en GitHub).

El preprocesamiento de imágenes incluye normalización y redimensionamiento uniforme.

El proyecto está estructurado para facilitar futuras ampliaciones, como añadir nuevos modelos o integraciones.

⚖️ Consideraciones éticas
El sistema es una herramienta de apoyo y no sustituye el diagnóstico profesional dermatológico.

Se recomienda validación clínica antes de su uso en contextos médicos reales.

Se respeta la privacidad y confidencialidad de los datos e imágenes utilizadas.

👥 Autores
Javier Silva

Juan Rojas

Nicolás Aros

