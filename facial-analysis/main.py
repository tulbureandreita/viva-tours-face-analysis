import cv2
import argparse
import warnings
import numpy as np
import os
from models import SCRFD, Attribute
from utils.helpers import Face, draw_face_info

warnings.filterwarnings("ignore")
run_number = 7

def load_models(detection_model_path: str, attribute_model_path: str):
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
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    return detection_model, attribute_model


def inference_image(detection_model, attribute_model, frame, save_output):
    """Processes a single image for face detection and attributes.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        image_path (str): Path to the input image.
        save_output (str): Path to save the output image.
    """

    if frame is None:
        print("Failed to load image")
        return

    face = process_frame(detection_model, attribute_model, frame)
    if save_output:
        cv2.imwrite(save_output, frame)
    if face is not None:
        x1, y1, x2, y2 = map(int, face.bbox)
        width = x2 - x1
        height = y2 - y1
        print(width, height)
        cropped = frame[y1:y2, x1:x2]
        save_output = save_output.removesuffix(".jpg")
        cv2.imwrite(save_output+"_crop.jpg", cropped)
        print(face.bbox)



def process_frame(detection_model, attribute_model, frame):
    """Detects faces and attributes in a frame and draws the information.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        frame (np.ndarray): The image frame to process.
    """
    boxes_list, points_list = detection_model.detect(frame)

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        gender = attribute_model.get(frame, bbox)
        face = Face(kps=keypoints, bbox=bbox, gender=gender)
        draw_face_info(frame, face)

        return face


def run_face_analysis(detection_weights, attribute_weights, frame, save_output=None):
    """Runs face detection on the given input source."""
    detection_model, attribute_model = load_models(detection_weights, attribute_weights)

    inference_image(detection_model, attribute_model, frame, save_output)

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
    """Main function to run face detection from command line."""
    parser = argparse.ArgumentParser(description="Run face detection on an image or video")
    parser.add_argument(
        '--detection-weights',
        type=str,
        default="weights/det_10g.onnx",
        help='Path to the detection model weights file'
    )
    parser.add_argument(
        '--attribute-weights',
        type=str,
        default="weights/genderage.onnx",
        help='Path to the attribute model weights file'
    )
    folder_name = "assets/dev-images/"
    args = parser.parse_args()
    for filename in os.listdir(folder_name):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        image_path = folder_name + filename
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Could not load {image_path}.jpg")
        MAX_SIDE_FOR_DETECTION = 1280
        small_img, sx, sy = resize_with_aspect(img=orig_img, max_side=MAX_SIDE_FOR_DETECTION)
        filename = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace(".bmp", "")
        if not os.path.exists(f"results/{run_number}"):
            os.makedirs(f"results/{run_number}")
        file_name_output_img = f"results/{run_number}/{filename}_SCRFD.jpg"
        run_face_analysis(args.detection_weights, args.attribute_weights, small_img, file_name_output_img)


if __name__ == "__main__":
    main()
