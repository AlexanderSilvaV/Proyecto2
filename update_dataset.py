import os
import pandas as pd

# Rutas
dataset_folder = r"C:\Users\j.silva\Documents\proyecto\Alopecia_Project\alopecia_project\dataset\bald_people"
image_folder = os.path.join(dataset_folder, 'images')
csv_path = os.path.join(dataset_folder, 'bald_people.csv')

# Cargar CSV
df = pd.read_csv(csv_path)

# --- LIMPIEZA DE IMÁGENES CORRUPTAS (detectadas previamente y listadas) ---
# Aquí debes reemplazar esta lista por las imágenes que tu CNN detecte como corruptas
# Por ejemplo, esta lista podría ser pasada desde tu clase Dataset o un archivo de texto.
corrupted_images_detected = [
    # Ejemplo de rutas absolutas detectadas como corruptas:
    # r"C:\Users\j.silva\Documents\proyecto\Alopecia_Project\alopecia_project\dataset\bald_people\images\imagen_corrupta.jpg"
]

if corrupted_images_detected:
    print(f"Eliminando {len(corrupted_images_detected)} imágenes corruptas y actualizando CSV...")

    # Convertir rutas absolutas a rutas relativas usadas en CSV ("images/filename.ext")
    corrupted_rel_paths = ['images/' + os.path.basename(p) for p in corrupted_images_detected]

    # Eliminar filas del CSV que correspondan a esas imágenes corruptas
    df = df[~df['images'].isin(corrupted_rel_paths)]

    # Guardar CSV actualizado sin índice extra
    df.to_csv(csv_path, index=False)

    # Eliminar archivos corruptos del disco
    for file_path in corrupted_images_detected:
        try:
            os.remove(file_path)
            print(f"Archivo eliminado: {file_path}")
        except Exception as e:
            print(f"No se pudo eliminar {file_path}: {e}")

else:
    print("No se detectaron imágenes corruptas para eliminar.")

# --- DETECTAR NUEVAS IMÁGENES Y AGREGARLAS AL CSV ---
# Obtener conjunto de imágenes en CSV
images_in_csv = set(df['images'].tolist())  # ej: 'images/eu.xxx.jpg'

# Obtener conjunto de imágenes en carpeta (solo nombres de archivo)
images_in_folder = set(os.listdir(image_folder))

# Filtrar solo imágenes válidas
valid_ext = {'.jpg', '.jpeg', '.png'}
images_in_folder = {f for f in images_in_folder if os.path.splitext(f)[1].lower() in valid_ext}

# Crear conjunto de imágenes con path relativo para comparar con CSV
images_in_folder_rel = {'images/' + f for f in images_in_folder}

# Encontrar imágenes nuevas que están en carpeta pero no en CSV
new_images = images_in_folder_rel - images_in_csv

print(f"Imágenes nuevas detectadas: {len(new_images)}")

# Asignar etiqueta fija (modifica si quieres otra)
new_label = "type_7"

# Crear lista para nuevas filas
new_rows = [{'images': img_path, 'type': new_label} for img_path in new_images]

# Crear dataframe nuevo para las filas nuevas
df_new = pd.DataFrame(new_rows)

# Concatenar y guardar CSV actualizado sin índice extra
df_updated = pd.concat([df, df_new], ignore_index=True)
df_updated.to_csv(csv_path, index=False)

print(f"CSV actualizado con {len(new_rows)} imágenes nuevas.")
