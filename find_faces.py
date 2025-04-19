import cv2
import dlib

# Load the dlib face detector
face_detector = dlib.get_frontal_face_detector()

def find_faces_in_image(image_path, padding_ratio=0.3):
    """
    Detect faces using dlib in the given image and return images with padding.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    face_images = []
    height, width = image.shape[:2]

    for face in faces:
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()

        # Calculate padding based on face size
        face_width = right - left
        face_height = bottom - top
        pad_w = int(face_width * padding_ratio)
        pad_h = int(face_height * padding_ratio)

        # Apply padding and ensure bounds are within the image
        new_left = max(left - pad_w, 0)
        new_top = max(top - pad_h, 0)
        new_right = min(right + pad_w, width)
        new_bottom = min(bottom + pad_h, height)

        face_image = image[new_top:new_bottom, new_left:new_right]
        face_images.append(face_image)

    return face_images

def find_faces_in_image_web(image_path, padding_ratio=0.3):
    """
    Detect faces using dlib in the given image and return images with padding.
    """
    # Read the image (BGR color space)
    image = cv2.imread(image_path)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in grayscale image
    faces = face_detector(gray)

    face_images = []
    height, width = image.shape[:2]

    for face in faces:
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()

        # Calculate padding based on face size
        face_width = right - left
        face_height = bottom - top
        pad_w = int(face_width * padding_ratio)
        pad_h = int(face_height * padding_ratio)

        # Apply padding and ensure bounds are within the image
        new_left = max(left - pad_w, 0)
        new_top = max(top - pad_h, 0)
        new_right = min(right + pad_w, width)
        new_bottom = min(bottom + pad_h, height)

        # Extract face region with color (from the original image)
        face_image = image[new_top:new_bottom, new_left:new_right]
        
        # Check if the face_image is still in color (3 channels)
        if face_image.shape[-1] != 3:
            print("Warning: Face image has lost color.")
        else:
            face_images.append(face_image)

    return face_images
