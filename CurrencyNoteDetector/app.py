from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load currency note templates
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

template_folder = "notes"
templates = {}

for filename in os.listdir(template_folder):
    if filename.lower().endswith((".jpg", ".jpeg")):
        value = filename.split(".")[0]
        path = os.path.join(template_folder, filename)
        img = cv2.imread(path, 0)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            templates[value] = (img, kp, des)

@app.route('/detect', methods=['POST'])
def detect_note():
    file = request.files.get('image')
    if not file:
        return jsonify({"result": "No file uploaded"})

    file_path = "temp.jpg"
    file.save(file_path)

    test_img = cv2.imread(file_path)
    if test_img is None:
        return jsonify({"result": "Failed to read image"})

    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    best_match = "Unknown"
    max_good_matches = 0

    for value, (tmpl_img, kp_tmpl, des_tmpl) in templates.items():
        if des_test is None or des_tmpl is None:
            continue

        matches = bf.match(des_tmpl, des_test)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50]

        if len(good_matches) > max_good_matches and len(good_matches) > 15:
            max_good_matches = len(good_matches)
            best_match = f"INR {value}"

    os.remove(file_path)
    return jsonify({"result": best_match})

if __name__ == '__main__':
    app.run(debug=True)
