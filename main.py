import cv2
import numpy as np
import os
from find_faces import find_faces_in_image
from facerecog import get_face_embeddings, compare_faces
import pymongo
from bson.binary import Binary
from pymongo import MongoClient

from config import MONGO_URI

# Initialize MongoDB connection
client = MongoClient(MONGO_URI)
db = client['face_recognition_db']
collection = db['faces']

def store_embedding_in_db(image_path, image_name):
    faces = find_faces_in_image(image_path)
    
    for i, face in enumerate(faces):
        temp_file_path = f"temp_face_{i}.jpg"
        cv2.imwrite(temp_file_path, face)

        embeddings, _ = get_face_embeddings(temp_file_path)

        if embeddings:
            face_data = {
                "name": image_name,
                "embedding": Binary(np.array(embeddings[0], dtype=np.float32).tobytes())
            }
            collection.insert_one(face_data)

        os.remove(temp_file_path)

    print(f"Stored embeddings for image {image_name}")

def combine_faces(uploaded_image, group_image, uploaded_face_location, group_face_location):
    top_u, right_u, bottom_u, left_u = (uploaded_face_location.top(), uploaded_face_location.right(),
                                        uploaded_face_location.bottom(), uploaded_face_location.left())
    top_g, right_g, bottom_g, left_g = (group_face_location.top(), group_face_location.right(),
                                        group_face_location.bottom(), group_face_location.left())

    uploaded_face = uploaded_image[top_u:bottom_u, left_u:right_u]
    group_face = group_image[top_g:bottom_g, left_g:right_g]

    if uploaded_face is None or group_face is None or uploaded_face.size == 0 or group_face.size == 0:
        print("Error: One of the faces is invalid.")
        return None

    group_face_resized = cv2.resize(group_face, (uploaded_face.shape[1], uploaded_face.shape[0]))
    combined_face = np.hstack((uploaded_face, group_face_resized))

    height, width = group_image.shape[:2]
    result_image = np.zeros_like(group_image)

    y_offset = height // 4
    x_offset = width // 4

    result_image[y_offset:y_offset+combined_face.shape[0], x_offset:x_offset+combined_face.shape[1]] = combined_face
    return result_image

def find_matching_face(image_path, group_image_path):
    uploaded_face_encoding, uploaded_face_locations = get_face_embeddings(image_path)
    if not uploaded_face_encoding:
        print("No face detected in uploaded image")
        return

    print("Uploaded face embeddings found")
    stored_faces = collection.find()

    best_match = None
    best_similarity = float('inf')

    for face in stored_faces:
        stored_embedding = np.frombuffer(face['embedding'], dtype=np.float32)
        distances = compare_faces([stored_embedding], uploaded_face_encoding)
        distance = distances[0]

        if distance < best_similarity:
            best_similarity = distance
            best_match = face

    if not best_match:
        print("No match found")
        return

    print(f"Found a match with {best_match['name']} (Distance: {best_similarity:.2f})")

    group_image = cv2.imread(group_image_path)
    group_embeddings, group_face_locations = get_face_embeddings(group_image_path)
    if not group_face_locations:
        print("No faces detected in group photo")
        return

    if best_similarity < 0.6:
        matched_face_location = group_face_locations[0]  # assuming first match
        uploaded_face_location = uploaded_face_locations[0]

        uploaded_image = cv2.imread(image_path)

        # Highlight the matched face in group photo
        top, right, bottom, left = (matched_face_location.top(), matched_face_location.right(),
                                    matched_face_location.bottom(), matched_face_location.left())
        cv2.rectangle(group_image, (left, top), (right, bottom), (0, 0, 255), 3)  # red box

        combined_image = combine_faces(
            uploaded_image,
            group_image,
            uploaded_face_location,
            matched_face_location
        )

        if combined_image is not None:
            cv2.imshow("Uploaded Image", uploaded_image)
            cv2.imshow("Group Photo (Matched Face Highlighted)", group_image)
            cv2.imshow("Combined Face Comparison", combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    store_embedding_in_db('group_photo2.jpg', 'Group Photo')
    find_matching_face('upload1.jpg', 'group_photo2.jpg')
