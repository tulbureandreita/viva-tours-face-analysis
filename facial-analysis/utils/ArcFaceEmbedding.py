import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
import cv2

# Aligns a face given 5 landmarks to a canonical 112×112 space
SRC_KEYPOINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)

def norm_crop(img: np.ndarray, kps: np.ndarray, image_size=(112, 112)) -> np.ndarray:
    """
    Warp-align the face in `img` using 5-point landmarks `kps` to a canonical image of size `image_size`.
    `kps` shape should be (5,2), coordinates in the original image.
    """
    src = kps.astype(np.float32)
    dst = SRC_KEYPOINTS
    # Estimate similarity transform (rotation, scale, translation)
    tform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(
        src=img,  # source image
        M=tform,  # 2×3 affine matrix
        dsize=image_size,  # output size (w,h)
        flags=cv2.INTER_LINEAR,  # interpolation method
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),  # fill with black in B, G, R channels
    )
    return aligned


@dataclass
class ArcFaceEmbedding:
    def __init__(self, onnx_path: str, input_size=(112,112)):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_shape = input_size

    def get(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        # 1) Align & crop the face patch to 112×112 using landmarks kps (5 pts)
        face = norm_crop(img, kps, image_size=self.input_shape)
        # 2) Preprocess: BGR→RGB, float32, transpose, add batch dim, normalize
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(face, (2,0,1))[None, ...]
        # 3) Inference
        embedding = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0].flatten()
        # 4) ℓ₂-normalize
        return embedding / np.linalg.norm(embedding)
