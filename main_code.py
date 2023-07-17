#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:36:24 2023

@author: maxime.pariente
"""

import os
from PIL import Image
from collections import namedtuple
import cv2
import numpy as np
from skimage import morphology

# Définition de la classe INFRA10Class
INFRA10Class = namedtuple('CityscapesClass', ['name', 'train_id', 'category', 'category_id',
                                              'has_instances', 'ignore_in_eval', 'color', 'grey'])

# Définition des classes
classes = [
    INFRA10Class('road',                 0, 'flat', 1, False, False, (224, 92, 94), (0, 0, 0)),
    INFRA10Class('sidewalk',             1, 'flat', 1, False, False, (98, 229, 212), (1, 1, 1)),
    INFRA10Class('building',             2, 'construction', 2, False, False, (75, 213, 234), (2, 2, 2)),
    INFRA10Class('wall',                 3, 'construction', 2, False, False, (42, 186, 83), (3, 3, 3)),
    INFRA10Class('fence',                4, 'construction', 2, False, False, (65, 255, 12), (4, 4, 4)),
    INFRA10Class('pole',                 5, 'object', 3, False, False, (46, 181, 211), (5, 5, 5)),
    INFRA10Class('traffic light',        6, 'object', 3, False, False, (38, 173, 42), (6, 6, 6)),
    INFRA10Class('traffic sign',         7, 'object', 3, False, False, (237, 61, 222), (7, 7, 7)),
    INFRA10Class('vegetation',           8, 'nature', 4, False, False, (122, 234, 2), (8, 8, 8)),
    INFRA10Class('terrain',              9, 'nature', 4, False, False, (86, 244, 247), (9, 9, 9)),
    INFRA10Class('sky',                  10, 'sky', 5, False, False, (87, 242, 87), (10, 10, 10)),
    INFRA10Class('person',               11, 'human', 6, True, False, (33, 188, 119), (11, 11, 11)),
    INFRA10Class('rider',                12, 'human', 6, True, False, (216, 36, 186), (12, 12, 12)),
    INFRA10Class('car',                  13, 'vehicle', 7, True, False, (224, 172, 51), (13, 13, 13)),
    INFRA10Class('truck',                14, 'vehicle', 7, True, False, (232, 196, 97), (14, 14, 14)),
    INFRA10Class('bus',                  15, 'vehicle', 7, True, False, (0, 137, 150), (15, 15, 15)),
    INFRA10Class('train',                16, 'vehicle', 7, True, False, (97, 232, 187), (16, 16, 16)),
    INFRA10Class('motorcycle',           17, 'vehicle', 7, True, False, (239, 107, 197), (17, 17, 17)),
    INFRA10Class('bicycle',              18, 'vehicle', 7, True, False, (149, 15, 252), (18, 18, 18)),
    INFRA10Class('unlabeled',            255, 'void', 0, False, True, (206, 140, 26), (255, 255, 255)),
]

def process_image(input_path, output_path):
    # Chargement de l'image d'entrée
    image = Image.open(input_path)
    
    # Conversion de l'image en mode RGBA si elle ne l'est pas déjà
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Parcours de chaque pixel de l'image
    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b, a = pixels[i, j]
            
            # Recherche de la classe correspondante à la couleur du pixel
            pixel_class = None
            for class_obj in classes:
                if class_obj.color == (r, g, b):
                    pixel_class = class_obj
                    break
            
            # Vérification si la classe correspond à 'unlabeled' pour mettre à jour la couleur
            if pixel_class is not None and pixel_class.name == 'unlabeled':
                pixels[i, j] = (1, 1, 1, a)  # Changement de couleur en blanc (255, 255, 255)
            else:
                pixels[i, j] = (0, 0, 0, a)  # Changement de couleur en noir (0, 0, 0)
    
    # Enregistrement de l'image modifiée
    image.save(output_path, 'PNG')


# Dossier contenant les fichiers d'entrée
input_folder = '/home/poc2014/dataset/temp/INFRA10/semantic_segmentation_truth/val/Massy/'
# Dossier de sortie pour les images modifiées
output_folder = '/Users/maxime.pariente/U2IS/testsortie'

# Parcourir les fichiers .png dans le dossier d'entrée
for filename in os.listdir(input_folder):
    if filename.endswith('.png') and not filename.endswith("labelIds.png"):
        # Chemin complet du fichier d'entrée
        input_path = os.path.join(input_folder, filename)
        # Chemin complet du fichier de sortie
        output_path = os.path.join(output_folder, filename)
        
        # Appeler la fonction pour traiter l'image et enregistrer l'image modifiée
        process_image(input_path, output_path)

        # Charger l'image modifiée
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

        a = np.array(mask, bool)
        ims = morphology.remove_small_objects(a, 725760)

        height, width = ims.shape

        for h in range(height):
            for w in range(width):
                if (image[h, w] == [206, 140, 26]).any() and (ims[h, w] == [1, 1, 1]).any():
                    image[h, w] = [26, 140, 206]
                    print(h, w)

        # Sauvegarder l'image modifiée
        modified_output_path = os.path.join(output_folder, 'modified_' + filename)
        cv2.imwrite(modified_output_path, image)
        #cv2.imwrite("/Users/maxime.pariente/U2IS/microdatabase/modified_image.png", image)

        
        
        
        
        
        
