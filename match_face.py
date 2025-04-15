import os
import cv2
import numpy as np
from facerecog import get_face_embeddings
from find_faces import find_faces_in_image  # <-- Import your face extractor
from plot_similarity import plot_face_similarity


STORAGE_DIR = "storage"
THRESHOLD = 0.6

def find_matching_face(uploaded_image_path):
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
        
        #plot the graph
        plot_face_similarity(uploaded_embedding[0], best_stored_embedding)


        # Get face (with padding) from uploaded image
        uploaded_faces = find_faces_in_image(uploaded_image_path)
        if not uploaded_faces:
            print("Could not extract face from uploaded image.")
            return
        uploaded_face_only = uploaded_faces[0]  # Assuming one face

        # Load matched image and draw a rectangle
        matched_image = cv2.imread(best_match_path)
        top, right, bottom, left = (best_match_location.top(), best_match_location.right(),
                                    best_match_location.bottom(), best_match_location.left())
        cv2.rectangle(matched_image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Resize matched image to same height as face image
        face_height = uploaded_face_only.shape[0]
        scale = face_height / matched_image.shape[0]
        new_width = int(matched_image.shape[1] * scale)
        matched_resized = cv2.resize(matched_image, (new_width, face_height))

        # Combine face-only and highlighted full image
        combined = np.hstack((uploaded_face_only, matched_resized))

        cv2.imshow("Uploaded Face vs Highlighted Match", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No good match found.")
