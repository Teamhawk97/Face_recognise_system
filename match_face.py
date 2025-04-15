import os
import cv2
import numpy as np
from facerecog import get_face_embeddings

STORAGE_DIR = "storage"
THRESHOLD = 0.6

def find_matching_face(uploaded_image_path):
    uploaded_embedding, uploaded_locations = get_face_embeddings(uploaded_image_path)
    if not uploaded_embedding:
        print("No face detected in uploaded image")
        return

    print("Uploaded face embeddings found")

    best_match_path = None
    best_distance = float('inf')

    for filename in os.listdir(STORAGE_DIR):
        if filename.endswith(".jpg"):
            stored_path = os.path.join(STORAGE_DIR, filename)
            stored_embedding, _ = get_face_embeddings(stored_path)

            if stored_embedding:
                distance = np.linalg.norm(uploaded_embedding[0] - stored_embedding[0])
                if distance < best_distance:
                    best_distance = distance
                    best_match_path = stored_path

    if best_distance < THRESHOLD and best_match_path:
        print(f"Match found with: {best_match_path} (Distance: {best_distance:.2f})")

        uploaded_image = cv2.imread(uploaded_image_path)
        matched_image = cv2.imread(best_match_path)

        matched_image = cv2.resize(matched_image, (uploaded_image.shape[1], uploaded_image.shape[0]))
        combined = np.hstack((uploaded_image, matched_image))

        cv2.imshow("Uploaded vs Matched", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No good match found.")
