import cv2
import base64
import os

def img_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def generate_html_report(uploaded_face_img, matched_face_img, person_data, similarity_plot_path, output_path="face_match_report.html"):
    uploaded_b64 = img_to_base64(uploaded_face_img)
    matched_b64 = img_to_base64(matched_face_img)

    with open(similarity_plot_path, "rb") as plot_file:
        plot_b64 = base64.b64encode(plot_file.read()).decode('utf-8')

    bio_info_html = ""
    for key, value in person_data.items():
        if key not in ["_id", "face_image"]:
            bio_info_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"

    html_content = f"""
    <html>
    <head><title>Face Match Report</title></head>
    <body style="font-family: Arial;">
        <h2>✅ Face matched with a person from the database</h2>

        <h3>👤 Bio Data:</h3>
        <table border="1" cellpadding="8" cellspacing="0">
            {bio_info_html}
        </table>

        <h3>📷 Uploaded Face vs Matched Face</h3>
        <div style="display: flex;">
            <div><img src="data:image/jpeg;base64,{uploaded_b64}" width="200"/></div>
            <div style="margin: 0 20px;"><img src="data:image/jpeg;base64,{matched_b64}" width="200"/></div>
        </div>

        <h3>📊 Similarity Plot</h3>
        <img src="data:image/png;base64,{plot_b64}" width="400"/>

    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"✅ HTML report generated at: {output_path}")
