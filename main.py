from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import numpy as np
import cv2
import mediapipe as mp

app = FastAPI(title="Face Validation (MediaPipe)")
app.add_middleware(
    CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# MediaPipe utilities
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398]
NOSE_IDX = [1, 4]
CHIN_IDX = [152]

# Validation thresholds
MIN_FACE_WIDTH_RATIO = 0.22
MIN_FACE_HEIGHT_RATIO = 0.22
EDGE_MARGIN = 0.06
MAX_NOD_TILT_RATIO = 0.22

class ValidateResult(BaseModel):
    ok: bool
    reasons: List[str] = []
    face_box: Optional[Dict[str, Any]] = None
    landmarks_found: Optional[Dict[str, int]] = None
    nose_offset_ratio: Optional[float] = None

def _read_image_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img

def _read_image_from_base64(data_url_or_b64: str) -> np.ndarray:
    if data_url_or_b64.startswith("data:"):
        header, b64 = data_url_or_b64.split(",", 1)
    else:
        b64 = data_url_or_b64
    b = base64.b64decode(b64)
    return _read_image_from_bytes(b)

def _get_largest_detection(detections):
    best = None
    for det in detections:
        rb = det.location_data.relative_bounding_box
        xmin = rb.xmin
        ymin = rb.ymin
        w = rb.width
        h = rb.height
        area = w * h
        if (best is None) or (area > best[0]):
            best = (area, xmin, ymin, w, h)
    if best is None:
        return None
    _, xmin, ymin, w, h = best
    return (xmin, ymin, w, h)

def _normalize_bbox(xmin, ymin, w, h, img_w, img_h):
    ax = int(max(0, xmin) * img_w)
    ay = int(max(0, ymin) * img_h)
    aw = int(min(1.0, w) * img_w)
    ah = int(min(1.0, h) * img_h)
    return ax, ay, aw, ah

def _check_landmarks_presence(landmarks, idxs) -> int:
    count = 0
    for i in idxs:
        lm = landmarks[i]
        if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            count += 1
    return count

def validate_face_image(img: np.ndarray) -> ValidateResult:
    h, w, _ = img.shape
    reasons = []
    try:
        # 1) Face detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as detector:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det_results = detector.process(rgb)
            if not det_results.detections:
                return ValidateResult(ok=False, reasons=["No face detected"])
            sel = _get_largest_detection(det_results.detections)
            if sel is None:
                return ValidateResult(ok=False, reasons=["No face bounding box found"])
            rel_x, rel_y, rel_w, rel_h = sel
            ax, ay, aw, ah = _normalize_bbox(rel_x, rel_y, rel_w, rel_h, w, h)

        # 2) Check bounding box cropping
        margin_x = int(w * EDGE_MARGIN)
        margin_y = int(h * EDGE_MARGIN)
        if ax < margin_x or ay < margin_y or (ax + aw) > (w - margin_x) or (ay + ah) > (h - margin_y):
            reasons.append("Face too close to image edge or cropped. Reposition the camera.")

        # 3) Check face size
        if aw < w * MIN_FACE_WIDTH_RATIO or ah < h * MIN_FACE_HEIGHT_RATIO:
            reasons.append("Face too small in frame. Move closer to camera.")

        # 4) Use face_mesh for landmarks
        with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1,
                                   min_detection_confidence=0.4, min_tracking_confidence=0.4) as mesh:
            mesh_res = mesh.process(rgb)
            if not mesh_res.multi_face_landmarks or len(mesh_res.multi_face_landmarks) == 0:
                return ValidateResult(ok=False, reasons=["No face landmarks detected; try again with better lighting."])
            landmarks = mesh_res.multi_face_landmarks[0].landmark
            left_eye_count = _check_landmarks_presence(landmarks, LEFT_EYE_IDX)
            right_eye_count = _check_landmarks_presence(landmarks, RIGHT_EYE_IDX)
            nose_count = _check_landmarks_presence(landmarks, NOSE_IDX)
            chin_count = _check_landmarks_presence(landmarks, CHIN_IDX)
            lands_found = {
                "left_eye": left_eye_count,
                "right_eye": right_eye_count,
                "nose": nose_count,
                "chin": chin_count,
            }
            if left_eye_count < 4 or right_eye_count < 4:
                reasons.append("Both eyes not clearly visible. Center your face and ensure eyes are open.")
            if nose_count < 1:
                reasons.append("Nose tip not detected.")
            if chin_count < 1:
                reasons.append("Chin not detected; face might be cropped at bottom.")

            # 5) Approximate yaw
            nose_lm = landmarks[NOSE_IDX[0]]
            nose_x = nose_lm.x
            bbox_center_x = (rel_x + rel_w / 2.0)
            nose_offset_ratio = abs(nose_x - bbox_center_x) / rel_w if rel_w > 0 else 1.0
            if nose_offset_ratio > MAX_NOD_TILT_RATIO:
                reasons.append("Face turned away (too much yaw). Please face camera directly.")

            face_box = {"x": ax, "y": ay, "w": aw, "h": ah}
            return ValidateResult(
                ok=len(reasons) == 0,
                reasons=reasons,
                face_box=face_box,
                landmarks_found=lands_found,
                nose_offset_ratio=nose_offset_ratio
            )
    except Exception as e:
        print(f"Error during face validation: {e}")
        return ValidateResult(ok=False, reasons=[f"Error during validation: {str(e)}"])

@app.post("/validate-file", response_model=ValidateResult)
async def validate_file(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = _read_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    res = validate_face_image(img)
    return res

class Base64Payload(BaseModel):
    image: str

@app.post("/validate-base64", response_model=ValidateResult)
async def validate_base64(payload: Base64Payload):
    try:
        img = _read_image_from_base64(payload.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    res = validate_face_image(img)
    return res
