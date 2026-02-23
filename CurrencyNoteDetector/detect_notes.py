import cv2
import os
from tkinter import Tk, filedialog

# Set up ORB feature detector and matcher
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load templates from "notes/" folder
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

if not templates:
    print("No templates found in 'notes/' folder.")
    exit()

# Select test image
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Select Currency Note Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    print("No image selected.")
    exit()

test_img = cv2.imread(file_path)
if test_img is None:
    print("Failed to read the selected image.")
    exit()

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

print(f"Detected: {best_match}")

