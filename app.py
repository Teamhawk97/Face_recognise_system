from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import uuid
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from match_face import find_matching_face_db
from find_faces import find_faces_in_image_web
from DB_helper import check_in_DB, upload_new_person, upload_existing_face, get_people_with_faces, get_person_by_register_no
from bson import ObjectId
from pymongo import MongoClient

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
people_col = db['people']
faces_col = db['faces']

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = 'brothiswasasecretkeylol' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # ➤ Detect faces and crop them from the uploaded image
        faces = find_faces_in_image_web(file_path)

        if faces:
            # Check if there is more than one face
            if len(faces) > 1:
                # Only process and store faces if there are multiple faces
                temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_faces')
                os.makedirs(temp_dir, exist_ok=True)

                face_filenames = []  # Just the filenames for use in templates

                for i, face_img in enumerate(faces):
                    if isinstance(face_img, np.ndarray):
                        face_img = Image.fromarray(face_img)

                    face_filename = f"face_{uuid.uuid4().hex}.jpg"
                    face_path = os.path.join(temp_dir, face_filename)
                    face_img.save(face_path)

                    face_filenames.append(face_filename)  # Store only the filename

                # Store face filenames in session
                session['faces'] = face_filenames
                session['image_path'] = file_path

                return render_template('confirm_face.html', face_paths=face_filenames)

            else:
                # Match the single face with the database
                result = find_matching_face_db(file_path)
                if result is None:
                    return "No match found."
                
                # Return the result immediately without storing anything
                return render_template('result.html', person_data=result)

        return "No faces detected."

    return "File type not allowed"

@app.route('/confirm_selected_face', methods=['POST'])
def confirm_selected_face():
    data = request.get_json()
    selected_index = data.get('index')

    face_paths = session.get('faces', [])
    if selected_index is None or not (0 <= selected_index < len(face_paths)):
        return jsonify(success=False), 400

    selected_face_filename = face_paths[selected_index]
    selected_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_faces', selected_face_filename)

    # Match the face
    result = find_matching_face_db(selected_face_path)

    # # Delete the temporary face images after processing
    # temp_faces_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_faces')
    # for face_filename in face_paths:
    #     face_path = os.path.join(temp_faces_dir, face_filename)
    #     if os.path.exists(face_path):
    #         os.remove(face_path)

    if result is None:
        return "No match found."

    return render_template('result.html', person_data=result)

@app.route('/process_person_upload')
def process_person_upload():
    return render_template('select_person.html')

@app.route('/check_register_no', methods=['GET'])
def check_register_no():
    reg = request.args.get('reg')
    person = check_in_DB(reg)
    print(reg)
    print(bool(person))
    return jsonify({'exists': bool(person)})

@app.route('/new_person')
def new_person_page():
    reg = request.args.get('reg')
    return render_template('new_person.html', reg=reg)

# Route to render existing person image upload page
@app.route('/existing_person')
def existing_person_page():
    reg = request.args.get('reg')
    person = get_person_by_register_no(reg)
    if not person:
        flash("Person not found.")
        return redirect(url_for('index'))

    return render_template('existing_person.html', person=person)


@app.route('/submit_new_person', methods=['POST'])
def submit_new_person():
    UPLOAD_FOLDER = 'static\dbupload'
    # Get form data
    name = request.form['name']
    register_no = request.form['register_no']
    email = request.form['email']
    dob = request.form['dob']
    age = request.form['age']

    # Get the image file
    image_file = request.files['image']
    
    if image_file and image_file.filename != '':
        # Ensure upload folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Secure the filename
        filename = secure_filename(f"{register_no}_{image_file.filename}")
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the image
        image_file.save(image_path)

        # Call your db_helper function and pass the image path
        success = upload_new_person(
            name=name,
            register_no=register_no,
            email=email,
            dob=dob,
            age=age,
            image_path=image_path  # ✅ Pass the path, not the file
        )

        # Do something based on `success`
        if success:
            flash("New person uploaded successfully!")
            return redirect(url_for('index'))
            
        else:
            flash("Upload failed.")
            return redirect(url_for('index'))
            
    else:
        flash("No image file selected.")
        return redirect(url_for('index'))
        
# Handle form submission for existing person image
@app.route('/submit_existing_person', methods=['POST'])
def submit_existing_person():
    UPLOAD_FOLDER = 'static\dbupload'
    # Get the form data
    reg = request.form['register_no']
    
    # Get the image file
    image_file = request.files.get('image')

    if image_file and image_file.filename != '':
        # Ensure upload folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Secure the filename and add register number as part of the filename
        filename = secure_filename(f"{reg}_{image_file.filename}")
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the image to the static/uploads folder
        image_file.save(image_path)

        # Call your db_helper function and pass the image path
        success = upload_existing_face(reg, image_path)  # Pass the image path instead of the file

        # Provide feedback based on success or failure
        if success:
            flash("Face image added for existing person.")
            return redirect(url_for('index')) # Or wherever you want to redirect
        else:
            flash("Failed to upload face image.")
            return redirect(url_for('index'))
            
    else:
        flash("No image file selected.")
        return redirect(url_for('index'))
    
@app.route('/view_all')
def view_all():
    people_with_faces = get_people_with_faces()
    return render_template('view_all.html', people=people_with_faces)
        
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, threaded=False)