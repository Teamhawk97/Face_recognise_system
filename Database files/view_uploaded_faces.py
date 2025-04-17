import cv2
import numpy as np
from pymongo import MongoClient

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]

# Access the 'people' collection to get biodata
people_collection = db["people"]
# Access the 'faces' collection to get face embeddings and images
faces_collection = db["faces"]

# Fetch all documents from the 'people' collection
all_people = people_collection.find()

for person in all_people:
    try:
        # Directly access the fields in the schema
        name = person.get("name", "N/A")
        register_no = person.get("register_no", "N/A")
        dob = person.get("dob", "N/A")
        age = person.get("age", "N/A")
        email = person.get("email", "N/A")

        print("\n---")
        print("Name:", name)
        print("Reg No:", register_no)
        print("DOB:", dob)
        print("Age:", age)
        print("Email:", email)

        # Fetch faces for the current person based on person_id
        person_id = person["_id"]
        faces = faces_collection.find({"person_id": person_id})

        for i, face in enumerate(faces):
            face_bytes = face.get("face_image")
            if not face_bytes:
                print("No image data found.")
                continue

            # Decode image from binary
            image_array = np.frombuffer(face_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                print("❌ Could not decode face image.")
                continue

            cv2.imshow(f"{name} - Face {i+1}", image)
            print("👉 Press any key to continue to the next image...")
            cv2.waitKey(0)  # Wait for any key press
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing person {person['_id']}: {str(e)}")
