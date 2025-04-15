import os
import cv2
import numpy as np
from pymongo import MongoClient
from facerecog import get_face_embeddings
from find_faces import find_faces_in_image

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["faces"]

# Ask user for input: n = new, e = existing
choice = input("Upload for (n) New person or (e) Existing person? [n/e]: ").strip().lower()

if choice == "e":
    reg_no = input("Enter the Register Number of the person: ").strip()
    person_document = collection.find_one({"biodata.register_no": reg_no})

    if not person_document:
        print(f"No person found with register number: {reg_no}")
        exit(1)
    else:
        BIO_DATA = person_document["biodata"]
        print(f"Found person: {BIO_DATA['name']} ({reg_no})")
        faces_data = person_document.get("faces", [])

elif choice == "n":
    name = input("Enter Name: ")
    reg_no = input("Enter Register Number: ")
    dob = input("Enter Date of Birth (YYYY-MM-DD): ")
    age = input("Enter Age: ")
    email = input("Enter Email: ")

    BIO_DATA = {
        "name": name,
        "register_no": reg_no,
        "dob": dob,
        "age": age,
        "email": email
    }

    person_document = None
    faces_data = []
else:
    print("Invalid choice. Please enter 'n' for new or 'e' for existing.")
    exit(1)

# Folder containing images
IMAGE_FOLDER = "face_upload_image"

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        print(f"\nProcessing: {image_path}")

        embeddings, locations = get_face_embeddings(image_path)
        if not embeddings or not locations:
            print(f"No face detected in {filename}")
            continue

        faces = find_faces_in_image(image_path)

        # Confirm if multiple faces found
        if len(faces) > 1:
            print(f"Multiple faces detected in {filename}")
            for i, face_img in enumerate(faces):
                preview = cv2.resize(face_img, (200, 200))
                cv2.imshow(f"Face {i+1}", preview)
            cv2.waitKey(1)
            confirm = input(f"Do you want to upload all faces from {filename}? (y/n): ")
            cv2.destroyAllWindows()
            if confirm.lower() != 'y':
                print(f"Skipping {filename}")
                continue

        # Add faces
        for emb, face_img in zip(embeddings, faces):
            success, encoded_image = cv2.imencode('.jpg', face_img)
            if not success:
                print(f"Failed to encode face image from {filename}")
                continue

            face_document = {
                "embedding": emb.tolist(),
                "face_image": encoded_image.tobytes(),
                "source_image": filename
            }
            faces_data.append(face_document)

# Save to MongoDB
if person_document:
    collection.update_one({"biodata.register_no": BIO_DATA["register_no"]}, {"$set": {"faces": faces_data}})
    print(f"\n✅ Updated {BIO_DATA['name']} with new faces.")
else:
    new_document = {
        "biodata": BIO_DATA,
        "faces": faces_data
    }
    collection.insert_one(new_document)
    print(f"\n✅ Inserted new person {BIO_DATA['name']} with faces.")

print("\n✅ Upload complete.")
