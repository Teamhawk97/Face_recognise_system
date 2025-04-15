import os
import cv2
from find_faces import find_faces_in_image

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def store_faces_from_group_photo(image_path):
    faces = find_faces_in_image(image_path)

    for i, face in enumerate(faces):
        filename = os.path.join(STORAGE_DIR, f"face_{i}.jpg")
        cv2.imwrite(filename, face)

    print(f"Stored {len(faces)} face(s) from {image_path} into '{STORAGE_DIR}'")
