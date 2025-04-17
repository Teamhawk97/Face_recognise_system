import cv2
import base64
import os
import webbrowser

def img_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def generate_html_report(uploaded_face_img, matched_face_img, person_data, similarity_plot_path,
                         template_path="result.html", output_path="result.html"):
    uploaded_b64 = img_to_base64(uploaded_face_img)
    matched_b64 = img_to_base64(matched_face_img)

    with open(similarity_plot_path, "rb") as plot_file:
        plot_b64 = base64.b64encode(plot_file.read()).decode('utf-8')

    bio_info_html = ""
    for key, value in person_data.items():
        if key not in ["_id", "face_image"]:
            bio_info_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>\n"

    with open(template_path, "r", encoding="utf-8") as template_file:
        template = template_file.read()

    html_filled = (template
                   .replace("{{BIO_DATA}}", bio_info_html)
                   .replace("{{UPLOADED_IMG}}", uploaded_b64)
                   .replace("{{MATCHED_IMG}}", matched_b64)
                   .replace("{{PLOT_IMG}}", plot_b64))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_filled)

    print(f"✅ HTML report updated at: {output_path}")
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


def prepare_report(uploaded_img, matched_img, plot_path):
    cv2.imwrite("static/results/uploaded.jpg", uploaded_img)
    cv2.imwrite("static/results/matched.jpg", matched_img)
    # You can also move or copy the plot image to the result folder
    if os.path.exists(plot_path):
        os.rename(plot_path, "static/results/plot.png")

