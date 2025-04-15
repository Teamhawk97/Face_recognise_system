import os
import cv2
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from facerecog import get_face_embeddings
from find_faces import find_faces_in_image

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
people_collection = db["people"]
faces_collection = db["faces"]

# Ask user for input: n = new, e = existing
choice = input("Upload for (n) New person or (e) Existing person? [n/e]: ").strip().lower()

if choice == "e":
    reg_no = input("Enter the Register Number of the person: ").strip()
    person = people_collection.find_one({"register_no": reg_no})

    if not person:
        print(f"No person found with register number: {reg_no}")
        exit(1)
    else:
        person_id = person["_id"]
        print(f"Found person: {person['name']} ({reg_no})")

elif choice == "n":
    name = input("Enter Name: ")
    reg_no = input("Enter Register Number: ")
    dob = input("Enter Date of Birth (YYYY-MM-DD): ")
    age = input("Enter Age: ")
    email = input("Enter Email: ")

    person_data = {
        "name": name,
        "register_no": reg_no,
        "dob": dob,
        "age": age,
        "email": email
    }
    person_id = people_collection.insert_one(person_data).inserted_id
    print(f"Inserted new person: {name} ({reg_no})")
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

        if len(faces) > 1:
            print(f"Multiple faces detected in {filename}")
            image_with_boxes = cv2.imread(image_path)

            for i, (face_img, location) in enumerate(zip(faces, locations)):
                # Draw rectangle around each face and label it
                x = location.left()
                y = location.top()
                w = location.right() - x
                h = location.bottom() - y

                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, f"Face {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show the image with face labels
            preview = cv2.resize(image_with_boxes, (800, 800))
            cv2.imshow(f"Faces in {filename}", preview)
            cv2.waitKey(1)

            for i, face_img in enumerate(faces):
                confirm = input(f"Does Face {i+1} belong to {person['name']}? (y/n): ")
                cv2.destroyAllWindows()

                if confirm.lower() == 'y':
                    # Process and store face if confirmed
                    success, encoded_image = cv2.imencode('.jpg', face_img)
                    if not success:
                        print(f"Failed to encode face image from {filename}")
                        continue

                    face_document = {
                        "person_id": person_id,
                        "embedding": embeddings[i].tolist(),
                        "face_image": encoded_image.tobytes(),
                        "source_image": filename
                    }
                    faces_collection.insert_one(face_document)
                    print(f"Face {i+1} added to the database.")
                else:
                    print(f"Skipping Face {i+1}.")
        else:
            # If only one face is detected, no confirmation is asked, just process it
            face_img = faces[0]
            location = locations[0]
            
            # Extract coordinates from the dlib.rectangle object
            x = location.left()
            y = location.top()
            w = location.right() - x
            h = location.bottom() - y
            
            image_with_box = cv2.imread(image_path)
            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the image with the highlighted face
            preview = cv2.resize(image_with_box, (800, 800))
            cv2.imshow(f"Face 1", preview)

            # No need for user confirmation since there's only one face
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

            cv2.destroyAllWindows()

print("\n✅ Upload complete.")
