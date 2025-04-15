from store_faces import store_faces_from_group_photo
from match_face import find_matching_face

if __name__ == "__main__":
    store_faces_from_group_photo('group_photo.jpg')   # Step 1: Store memory
    find_matching_face('upload1.jpg')                  # Step 2: Match against memory
