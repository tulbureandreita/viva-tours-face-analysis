import cv2
import warnings
import os
from models import SCRFD, Attribute
from utils.ArcFaceEmbedding import ArcFaceEmbedding
from utils.helpers import Face, draw_face_info
from utils.FaissRecognizer import FaissRecognizer


warnings.filterwarnings("ignore")
run_number = 9
FACE_RECOGNIZER_THRESHOLD = 0.506
MAX_SIDE_FOR_DETECTION = 2560

def load_models(detection_model_path: str, attribute_model_path: str, emb_model_path: str):
    """Loads the detection and attribute models.
    Args:
        detection_model_path (str): Path to the detection model file.
        attribute_model_path (str): Path to the attribute model file.
    Returns
        tuple: A tuple containing the detection model and the attribute model.

    """
    try:
        detection_model = SCRFD(model_path=detection_model_path)
        attribute_model = Attribute(model_path=attribute_model_path)
        emb_model = ArcFaceEmbedding(onnx_path=emb_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    return detection_model, attribute_model, emb_model


def inference_image(detection_model, attribute_model, emb_model, frame, save_output, face_recognizer):
    """
    Processes an image, finds all faces, and lets the recognizer handle the assignment.
    """
    if frame is None:
        print("Failed to load image")
        return

    # This part remains the same: get all detected faces
    faces = process_frame(detection_model=detection_model, attribute_model=attribute_model, frame=frame)

    if faces:
        # Create a list of embeddings for every detected face
        embeddings = [emb_model.get(img=frame, kps=face.kps) for face in faces]

        # Call the new, smarter recognition function with the list of embeddings
        person_id, best_similarity = face_recognizer.recognize_and_assign(embeddings)

        if person_id:
            print(f"Final assignment: Person ID {person_id} with similarity: {best_similarity:.4f}")
            face_recognizer.save_image(
                person_id=person_id,
                original_img=frame,
                basename=f"{person_id}_{best_similarity:.4f}"
            )

    # Save the output image with all face boxes drawn on it
    if save_output:
        cv2.imwrite(save_output, frame)

def process_frame(detection_model, attribute_model, frame):
    """Detects all faces in a frame, draws info, and returns a list of Face objects."""
    boxes_list, points_list = detection_model.detect(frame)

    # Create an empty list to store face objects
    detected_faces = []

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        gender = attribute_model.get(frame, bbox)
        face = Face(kps=keypoints, bbox=bbox, gender=gender)

        # Draw info for the current face
        draw_face_info(frame=frame, face=face)

        # Add the processed face to our list
        detected_faces.append(face)

    # Return the entire list of faces
    return detected_faces


def run_face_analysis(detection_weights, attribute_weights, emb_weights, frame, save_output, face_recognizer):
    """Runs face detection on the given input source."""
    detection_model, attribute_model, emb_model = load_models(detection_model_path=detection_weights, attribute_model_path=attribute_weights, emb_model_path=emb_weights)

    inference_image(detection_model=detection_model, attribute_model=attribute_model, emb_model=emb_model, frame=frame, save_output=save_output, face_recognizer=face_recognizer)

def resize_with_aspect(img, max_side):
    """
    Downscale img so that its longest side == max_side (if it’s bigger).
    Returns (resized_img, scale_x, scale_y).
    If img is already smaller, returns (img, 1.0, 1.0).
    """
    h, w = img.shape[:2]
    # If the image is already <= max_side in both dimensions, do nothing.
    if max(h, w) <= max_side:
        return img, 1.0, 1.0

    # Compute scale so that max(h,w)*scale = max_side
    if w >= h:
        scale = max_side / w
    else:
        scale = max_side / h

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    return resized, scale, scale

def main():
    detection_weights_path = "weights/scrfd_10g_bnkps.onnx"
    attribute_weights_path = "weights/genderage.onnx"
    emb_weights_path = "weights/glintr100.onnx"

    folder_name = "assets/dev-images/"
    dir_path = f"results/{run_number}/"
    if not os.path.exists(f"results/{run_number}"):
        os.makedirs(f"results/{run_number}")
    recognizer = FaissRecognizer(dir_path=dir_path, threshold=FACE_RECOGNIZER_THRESHOLD)
    for filename in os.listdir(folder_name):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        image_path = folder_name + filename
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Could not load {image_path}.jpg")
        small_img, sx, sy = resize_with_aspect(img=orig_img, max_side=MAX_SIDE_FOR_DETECTION)
        filename = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace(".bmp", "")
        print("_________________________________")
        print(filename)
        file_name_output_img = dir_path+f"{filename}_SCRFD.jpg"
        run_face_analysis(detection_weights=detection_weights_path, attribute_weights=attribute_weights_path, emb_weights=emb_weights_path, frame=small_img, save_output=file_name_output_img, face_recognizer=recognizer)


if __name__ == "__main__":
    main()
