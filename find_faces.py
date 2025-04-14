import cv2
import dlib

# Load the dlib face detector
face_detector = dlib.get_frontal_face_detector()

def find_faces_in_image(image_path):
    """
    Detect faces using dlib in the given image.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    face_images = []
    for face in faces:
        left, top, right, bottom = (face.left(), face.top(), face.right(), face.bottom())
        face_image = image[top:bottom, left:right]
        face_images.append(face_image)
    
    return face_images
