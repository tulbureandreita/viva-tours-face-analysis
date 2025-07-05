# In facial-analysis/api_main.py

import cv2
import numpy as np
import warnings
import os
from fastapi import FastAPI, File, UploadFile

# Import your existing classes and functions
from models import SCRFD, Attribute
from utils.ArcFaceEmbedding import ArcFaceEmbedding
from utils.FaissRecognizer import FaissRecognizer

warnings.filterwarnings("ignore")

# --- 1. Load Models and Recognizer on Startup ---
print("Loading models and Faiss Recognizer...")

DETECTION_WEIGHTS_PATH = "weights/det_10g.onnx"
ATTRIBUTE_WEIGHTS_PATH = "weights/genderage.onnx"
EMB_WEIGHTS_PATH = "weights/glintr100.onnx"
RESULTS_DIR = "results/api_runs/"

os.makedirs(RESULTS_DIR, exist_ok=True)

DETECTION_MODEL = SCRFD(model_path=DETECTION_WEIGHTS_PATH)
ATTRIBUTE_MODEL = Attribute(model_path=ATTRIBUTE_WEIGHTS_PATH)
EMB_MODEL = ArcFaceEmbedding(onnx_path=EMB_WEIGHTS_PATH)

FACE_RECOGNIZER = FaissRecognizer(dir_path=RESULTS_DIR, threshold=0.505)  # Use your tuned threshold

print("Models loaded successfully!")

# --- 2. Initialize FastAPI App ---
app = FastAPI(title="Face Recognition API")


# --- 3. Refactor your processing logic for best match ---
def process_image_for_best_match(image: np.ndarray):
    """
    Takes a single image, finds the single best face match, and returns that person's UUID.
    """
    boxes_list, points_list = DETECTION_MODEL.detect(image)

    best_overall_similarity = -1.0
    best_match_id = None

    if boxes_list is not None:
        all_embeddings = []
        # First, get embeddings for all detected faces
        for boxes, keypoints in zip(boxes_list, points_list):
            *bbox, conf_score = boxes
            embedding = EMB_MODEL.get(img=image, face_bbox=bbox)
            all_embeddings.append(embedding)

        # Now, find the best match among all embeddings
        if all_embeddings:
            # This logic is now inside your FaissRecognizer
            person_id, best_similarity = FACE_RECOGNIZER.recognize_and_assign(all_embeddings)
            best_match_id = person_id

            # Save the image to the best match's folder
            if best_match_id:
                base_filename = f"{best_match_id}_{best_similarity:.4f}"
                FACE_RECOGNIZER.save_image(person_id=best_match_id, original_img=image, basename=base_filename)

    return best_match_id


# --- 4. Define the API Endpoint ---
@app.post("/upload_image/", tags=["Face Recognition"])
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts an image file, finds the best face match,
    and returns the corresponding person's UUID.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    MAX_SIDE = 2560
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    # Process the image to find the single best match
    best_match_uuid = process_image_for_best_match(img)

    return {
        "filename": file.filename,
        "best_match_person_uuid": best_match_uuid
    }