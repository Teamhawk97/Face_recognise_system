from store_faces import store_faces_from_group_photo
from store_faces import clear_storage
from store_faces import load_faces_from_db
from match_face import find_matching_face

if __name__ == "__main__":
    #store_faces_from_group_photo('group_photo2.jpg')   # Step 1: Store memory
    load_faces_from_db()                               # Load stored faces
    find_matching_face('upload1.jpg')                  # Step 2: Match against memory
    clear_storage()                                    # Clear storage if needed
