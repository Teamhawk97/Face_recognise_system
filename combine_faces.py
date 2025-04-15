import numpy as np
import cv2

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

