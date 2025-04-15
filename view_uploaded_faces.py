import cv2
import numpy as np
from pymongo import MongoClient

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["faces"]

# Fetch all documents
all_people = collection.find()

for person in all_people:
    biodata = person["biodata"]
    print("\n---")
    print("Name:", biodata.get("name"))
    print("Reg No:", biodata.get("register_no"))
    print("DOB:", biodata.get("dob"))
    print("Age:", biodata.get("age"))
    print("Email:", biodata.get("email"))

    for i, face in enumerate(person.get("faces", [])):
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

        cv2.imshow(f"{biodata['name']} - Face {i+1}", image)
        print("👉 Press any key to continue to the next image...")
        cv2.waitKey(0)  # 🔁 Wait for any key press
        cv2.destroyAllWindows()
