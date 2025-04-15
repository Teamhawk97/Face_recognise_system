from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import base64
import os

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
people_collection = db["people"]
faces_collection = db["faces"]

def generate_html_report(person_id, matched_face_ids):
    person = people_collection.find_one({"_id": ObjectId(person_id)})
    if not person:
        print("Person not found.")
        return

    name = person.get("name", "")
    register_no = person.get("register_no", "")
    dob = person.get("dob", "")
    age = person.get("age", "")
    email = person.get("email", "")

    # Start building HTML content
    html_content = f"""
    <html>
    <head>
        <title>Face Recognition Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #333;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            table, th, td {{
                border: 1px solid #ccc;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
            }}
            .face-images {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-top: 20px;
            }}
            .face-images img {{
                width: 150px;
                border: 1px solid #ccc;
                padding: 5px;
                background-color: white;
            }}
        </style>
    </head>
    <body>
        <h1>Face Recognition Report</h1>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <table>
            <tr><th>Name</th><td>{name}</td></tr>
            <tr><th>Register No</th><td>{register_no}</td></tr>
            <tr><th>Date of Birth</th><td>{dob}</td></tr>
            <tr><th>Age</th><td>{age}</td></tr>
            <tr><th>Email</th><td>{email}</td></tr>
        </table>
        <h2>Matched Faces:</h2>
        <div class="face-images">
    """

    for face_id in matched_face_ids:
        face = faces_collection.find_one({"_id": ObjectId(face_id)})
        if not face:
            continue

        image_path = face.get("image_path")  # Assuming image is saved as file path
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                html_content += f'<img src="data:image/jpeg;base64,{encoded}" alt="Face {face_id}">'
        else:
            html_content += f'<p>Image not found for face ID: {face_id}</p>'

    html_content += """
        </div>
    </body>
    </html>
    """

    report_path = f"face_report_{person_id}.html"
    with open(report_path, "w") as file:
        file.write(html_content)

    print(f"HTML report saved as {report_path}")
