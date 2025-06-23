import os
import time
import json
import urllib.request
from selenium import webdriver


from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def download_images_bing_selenium(query, max_images=100, save_folder=None):
    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), "images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    options = Options()
    options.add_argument("--headless")  # Ejecutar sin ventana visible
    driver = webdriver.Chrome(options=options)  # Cambia si usas otro driver

    search_url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2"
    driver.get(search_url)

    image_count = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    while image_count < max_images:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # espera a que cargue

        thumbnails = driver.find_elements(By.CLASS_NAME, "iusc")
        print(f"Encontrados thumbnails: {len(thumbnails)}")
        
        for thumbnail in thumbnails[image_count:]:
            if image_count >= max_images:
                break
            m_json = thumbnail.get_attribute('m')
            info = json.loads(m_json)
            img_url = info.get('murl')
            try:
                ext = os.path.splitext(img_url)[1].split('?')[0]  # extensión limpia
                if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                    ext = '.jpg'
                img_path = os.path.join(save_folder, f"alopecia_{image_count}{ext}")
                urllib.request.urlretrieve(img_url, img_path)
                print(f"Descargada imagen {image_count + 1}: {img_url}")
                image_count += 1
            except Exception as e:
                print(f"Error al descargar imagen: {e}")

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("No hay más imágenes para cargar.")
            break
        last_height = new_height

    driver.quit()

if __name__ == "__main__":
    ruta_guardado = r"C:\Users\j.silva\Documents\proyecto\Alopecia_Project\alopecia_project\dataset\bald_people\images"
    download_images_bing_selenium("personas con alopecia", max_images=100, save_folder=ruta_guardado)
