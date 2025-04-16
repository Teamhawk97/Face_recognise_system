import os
import io
import cv2
import numpy as np
from PIL import Image
from pymongo import MongoClient
from bson import ObjectId
from facerecog import get_face_embeddings
from find_faces import find_faces_in_image
from plot_similarity import plot_face_similarity
from generate_report import generate_html_report

# Constants
STORAGE_DIR = "storage"
THRESHOLD = 0.6

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
faces_collection = db["faces"]
people_collection = db["people"]

def find_matching_face_local(uploaded_image_path):
    uploaded_embedding, uploaded_locations = get_face_embeddings(uploaded_image_path)
    if not uploaded_embedding or not uploaded_locations:
        print("No face detected in uploaded image")
        return

    print("Uploaded face embeddings found")

    best_match_path = None
    best_distance = float('inf')
    best_match_location = None
    best_stored_embedding = None

    for filename in os.listdir(STORAGE_DIR):
        if filename.endswith(".jpg"):
            stored_path = os.path.join(STORAGE_DIR, filename)
            stored_embedding, stored_locations = get_face_embeddings(stored_path)

            if stored_embedding and stored_locations:
                distance = np.linalg.norm(uploaded_embedding[0] - stored_embedding[0])
                if distance < best_distance:
                    best_distance = distance
                    best_match_path = stored_path
                    best_match_location = stored_locations[0]  # Assuming one face per image
                    best_stored_embedding = stored_embedding[0]

    if best_distance < THRESHOLD and best_match_path:
        print(f"Match found with: {best_match_path} (Distance: {best_distance:.2f})")
        plot_face_similarity(uploaded_embedding[0], best_stored_embedding)

        uploaded_faces = find_faces_in_image(uploaded_image_path)
        if not uploaded_faces:
            print("Could not extract face from uploaded image.")
            return
        uploaded_face_only = uploaded_faces[0]

        matched_image = cv2.imread(best_match_path)
        top, right, bottom, left = (best_match_location.top(), best_match_location.right(),
                                    best_match_location.bottom(), best_match_location.left())
        cv2.rectangle(matched_image, (left, top), (right, bottom), (0, 0, 255), 2)

        face_height = uploaded_face_only.shape[0]
        scale = face_height / matched_image.shape[0]
        new_width = int(matched_image.shape[1] * scale)
        matched_resized = cv2.resize(matched_image, (new_width, face_height))

        combined = np.hstack((uploaded_face_only, matched_resized))
        cv2.imshow("Uploaded Face vs Highlighted Match", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No good match found.")

def find_matching_face_db(uploaded_image_path):
    uploaded_embedding, uploaded_locations = get_face_embeddings(uploaded_image_path)
    if not uploaded_embedding or not uploaded_locations:
        print("No face detected in uploaded image")
        return

    print("Uploaded face embeddings found")

    best_match_face = None
    best_distance = float('inf')

    all_faces = list(faces_collection.find({}))

    for face in all_faces:
        stored_embedding = face.get("embedding")
        if not stored_embedding:
            continue

        distance = np.linalg.norm(uploaded_embedding[0] - stored_embedding)
        if distance < best_distance:
            best_distance = distance
            best_match_face = face

    if best_distance < THRESHOLD and best_match_face:
        print(f"Match found in database (Distance: {best_distance:.2f})")

        # Show similarity plot
        plot_face_similarity(uploaded_embedding[0], best_match_face["embedding"],output_path="plot_image")

        # Extract face from uploaded image
        uploaded_faces = find_faces_in_image(uploaded_image_path)
        if not uploaded_faces:
            print("Could not extract face from uploaded image.")
            return
        uploaded_face_only = uploaded_faces[0]

        # Get matched face image from binary data
        image_bytes = best_match_face.get("face_image")
        if not image_bytes:
            print("No image data found in database.")
            return

        # Convert binary to OpenCV format
        image_stream = io.BytesIO(image_bytes)
        pil_image = Image.open(image_stream).convert('RGB')
        matched_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Resize matched image to same height as face image
        face_height = uploaded_face_only.shape[0]
        scale = face_height / matched_image.shape[0]
        new_width = int(matched_image.shape[1] * scale)
        matched_resized = cv2.resize(matched_image, (new_width, face_height))

        # Combine face-only and matched image
        combined = np.hstack((uploaded_face_only, matched_resized))
        cv2.imshow("Uploaded Face vs Database Match", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 🔍 Get biodata using person_id
        person_id = best_match_face.get("person_id")
        if person_id:
            biodata = people_collection.find_one({"_id": person_id})
            if biodata:
                print("\n--- Matched Person Biodata ---")
                for key, value in biodata.items():
                    if key != "_id":
                        print(f"{key.capitalize()}: {value}")
                generate_html_report(
                    uploaded_face_img=uploaded_face_only,
                    matched_face_img=matched_resized,
                    person_data=biodata,
                    similarity_plot_path="plot_image.png",
                    output_path="face_match_report.html"
                    )
            else:
                print("No biodata found for matched person.")
        else:
            print("No person_id found in face data.")

    else:
        print("No good match found in database.")