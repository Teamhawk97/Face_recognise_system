from pymongo import MongoClient
from flask import jsonify
import cv2
import os
import base64
import numpy as np
from facerecog import get_face_embeddings
from find_faces import find_faces_in_image
from bson import ObjectId

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
people_collection = db["people"]
faces_collection = db["faces"]


def check_in_DB(reg):
    existing_person = people_collection.find_one({"register_no": reg})
    return bool(existing_person)

def upload_new_person(name, register_no, email, dob, age, image_path):
    # Save person biodata to MongoDB
    person_data = {
        "name": name,
        "register_no": register_no,
        "dob": dob,
        "age": age,
        "email": email
    }
    person_id = people_collection.insert_one(person_data).inserted_id
    print(f"Inserted new person: {name} ({register_no})")

    # Folder containing images
    IMAGE_FOLDER = "static\dbupload"

    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            print(f"\nProcessing: {image_path}")

            embeddings, locations = get_face_embeddings(image_path)
            if not embeddings or not locations:
                print(f"No face detected in {filename}")
                continue

            faces = find_faces_in_image(image_path)

            # Always assume one face
            face_img = faces[0]
            location = locations[0]

            # Extract coordinates from the dlib.rectangle object
            x = location.left()
            y = location.top()
            w = location.right() - x
            h = location.bottom() - y

            image_with_box = cv2.imread(image_path)
            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Encode and store face
            success, encoded_image = cv2.imencode('.jpg', face_img)
            if not success:
                print(f"Failed to encode face image from {filename}")
                continue

            face_document = {
                "person_id": person_id,
                "embedding": embeddings[0].tolist(),
                "face_image": encoded_image.tobytes(),
                "source_image": filename
            }
            faces_collection.insert_one(face_document)
            print(f"Face added to the database.")

    print("Uploading to DB:")

    # Delete all image files from the upload folder
    for filename in os.listdir(IMAGE_FOLDER):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    return True

def upload_existing_face(reg, image_file):
    person = people_collection.find_one({"register_no": reg})
    
    if not person:
        print(f"Person with register number {reg} not found.")
        return False
    
    # Folder containing images
    IMAGE_FOLDER = "static\dbupload"
    
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            print(f"\nProcessing: {image_path}")

            embeddings, locations = get_face_embeddings(image_path)
            if not embeddings or not locations:
                print(f"No face detected in {filename}")
                continue

            faces = find_faces_in_image(image_path)
            face_img = faces[0]
            location = locations[0]
            
            # Extract coordinates from the dlib.rectangle object
            x = location.left()
            y = location.top()
            w = location.right() - x
            h = location.bottom() - y
            
            image_with_box = cv2.imread(image_path)
            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            success, encoded_image = cv2.imencode('.jpg', face_img)
            if not success:
                print(f"Failed to encode face image from {filename}")
                continue

            face_document = {
                "person_id": person['_id'],
                "embedding": embeddings[0].tolist(),
                "face_image": encoded_image.tobytes(),
                "source_image": filename
            }
            faces_collection.insert_one(face_document)
            print(f"Face added to the database.")

    print(f"Adding face for: {reg}")

    # Delete all image files from the upload folder
    for filename in os.listdir(IMAGE_FOLDER):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    return True

def get_people_with_faces():
    people = list(people_collection.find())
    people_with_faces = []

    for person in people:
        try:
            name = person.get("name", "N/A")
            register_no = person.get("register_no", "N/A")
            dob = person.get("dob", "N/A")
            age = person.get("age", "N/A")
            email = person.get("email", "N/A")

            person_id = person["_id"]
            faces = faces_collection.find({"person_id": person_id})

            face_images = []
            for face in faces:
                face_bytes = face.get("face_image")
                if not face_bytes:
                    continue

                img_array = np.frombuffer(face_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                face_images.append(img_base64)

            people_with_faces.append({
                "name": name,
                "register_no": register_no,
                "dob": dob,
                "age": age,
                "email": email,
                "faces": face_images
            })

        except Exception as e:
            print(f"Error processing person {person.get('_id')}: {str(e)}")

    return people_with_faces
    
def get_person_by_register_no(register_no):
    return people_collection.find_one({'register_no': register_no})
