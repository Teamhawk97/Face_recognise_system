import os
import cv2
import numpy as np
from find_faces import find_faces_in_image
from pymongo import MongoClient

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def clear_storage():
    if not os.path.exists(STORAGE_DIR):
        print(f"⚠️ Directory '{STORAGE_DIR}' does not exist.")
        return

    removed_files = 0
    for filename in os.listdir(STORAGE_DIR):
        file_path = os.path.join(STORAGE_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            removed_files += 1

    print(f"🧹 Cleared {removed_files} file(s) from '{STORAGE_DIR}'")

def store_faces_from_group_photo(image_path):
    faces = find_faces_in_image(image_path)

    for i, face in enumerate(faces):
        filename = os.path.join(STORAGE_DIR, f"face_{i}.jpg")
        cv2.imwrite(filename, face)

    print(f"Stored {len(faces)} face(s) from {image_path} into '{STORAGE_DIR}'")

def load_faces_from_db():
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    client = MongoClient("mongodb://localhost:27017/")
    db = client["face_recognition"]
    collection = db["faces"]

    cursor = collection.find()

    count = 0
    for doc in cursor:
        name = doc.get("biodata", {}).get("name", "unknown")
        register_no = doc.get("biodata", {}).get("register_no", "unknown")
        faces = doc.get("faces", [])

        for i, face_entry in enumerate(faces):
            face_data = np.frombuffer(face_entry["face_image"], dtype=np.uint8)
            face_img = cv2.imdecode(face_data, cv2.IMREAD_COLOR)

            filename = os.path.join(STORAGE_DIR, f"{name}_{register_no}_face_{i+1}.jpg")
            cv2.imwrite(filename, face_img)
            count += 1

    print(f"✅ Saved {count} face image(s) from MongoDB into '{STORAGE_DIR}'")