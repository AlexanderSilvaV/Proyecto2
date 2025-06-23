import os
import shutil
import pandas as pd

# Ruta del CSV y las imágenes
csv_path = 'dataset/bald_people/bald_people.csv'
images_dir = 'dataset/bald_people/images'
output_dir = 'dataset/bald_people_imagefolder'

# Cargar el CSV
df = pd.read_csv(csv_path)

# Crear carpetas por clase y copiar imágenes
for _, row in df.iterrows():
    img_filename = os.path.basename(row['images'])  # ej: eu.123.jpg
    label = row['type']                             # ej: type_3

    source_path = os.path.join(images_dir, img_filename)
    target_dir = os.path.join(output_dir, label)
    target_path = os.path.join(target_dir, img_filename)

    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(source_path, target_path)

print("✅ Dataset preparado correctamente en:", output_dir)
