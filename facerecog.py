import dlib
import numpy as np
import cv2

# Load the pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Adjust path

# Load the face recognition model
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")  # Adjust path

def get_face_embeddings(image_path):
    """
    Extracts face embeddings using the dlib face recognition model.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    embeddings = []
    face_locations = []

    for face in faces:
        shape = shape_predictor(image, face)
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        embeddings.append(np.array(face_descriptor))
        face_locations.append(face)

    return embeddings, face_locations

def compare_faces(embeddings1, embeddings2):
    """
    Compare two sets of face embeddings using Euclidean distance.
    """
    distances = [np.linalg.norm(embedding1 - embedding2) for embedding1, embedding2 in zip(embeddings1, embeddings2)]
    return distances
