import os
from PIL import Image

def convert_to_grayscale(input_path, output_path):
    # Ouvrir l'image d'entrée
    with Image.open(input_path) as img:
        # Convertir l'image en niveaux de gris (L mode)
        img_gray = img.convert("L")
        
        # Sauvegarder l'image en niveaux de gris, en écrasant l'image d'origine
        img_gray.save(output_path)

def process_images_in_folder(folder_path):
    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        print(f"Le dossier '{folder_path}' n'existe pas.")
        return

    # Parcourir tous les fichiers du dossier
    for filename in os.listdir(folder_path):
        if filename.endswith("labelIds.png"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, filename)

            # Convertir l'image en niveaux de gris et la sauvegarder
            convert_to_grayscale(input_path, output_path)

            # Supprimer l'image d'origine
            os.remove(input_path)

            print(f"Image '{filename}' traitée et remplacée par l'image en niveaux de gris.")

# Exemple d'utilisation
input_folder_path = "/home/poc2014/dataset/temp/INFRA10/semantic_segmentation_truth/val/Massy"
process_images_in_folder(input_folder_path)
