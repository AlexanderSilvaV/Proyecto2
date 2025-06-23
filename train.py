from alopecia_app.cnn_model import train_model
import os

# Ruta base del proyecto
base_path = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas para CSV y carpeta de imágenes
csv_file = os.path.join(base_path, 'dataset', 'bald_people', 'bald_people.csv')
root_dir = os.path.join(base_path, 'dataset', 'bald_people')
model_path = os.path.join(base_path, 'alopecia_cnn.pth')

# Entrenamiento del modelo
train_model(
    csv_file=csv_file,
    root_dir=root_dir,
    num_epochs=10,
    batch_size=16,
    lr=0.005,
    model_path=model_path
)

