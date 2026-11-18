import cv2
import sys
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

image_name = sys.argv[1] if len(sys.argv) > 1 else "round.jpg"
IMAGE_PATH = SCRIPT_DIR / image_name

HAIR_MODEL_PATH = str(SCRIPT_DIR / "hair_segmenter.tflite")
FACE_MODEL_PATH = str(SCRIPT_DIR / "face_landmarker.task")

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Could not find {IMAGE_PATH}")

height, width = image.shape[:2]
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)


# Estimate the hairline from the hair segmentation mask
hair_options = vision.ImageSegmenterOptions(
    base_options=python.BaseOptions(model_asset_path=HAIR_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    output_category_mask=True,
    output_confidence_masks=False,
)
with vision.ImageSegmenter.create_from_options(hair_options) as segmenter:
    hair_result = segmenter.segment(mp_image)
    mask = hair_result.category_mask.numpy_view().copy()

if mask.shape != (height, width):
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

hair_mask = (mask == 1).astype(np.uint8) * 255


# Detect face landmarks
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
with vision.FaceLandmarker.create_from_options(face_options) as landmarker:
    face_result = landmarker.detect(mp_image)

if not face_result.face_landmarks:
    raise RuntimeError("No face detected.")

landmarks = face_result.face_landmarks[0]


def point(index):
    """Return a landmark as (x, y) pixel coordinates."""
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def distance(index_a, index_b):
    """Return the straight-line distance between two landmarks in pixels."""
    return float(np.linalg.norm(point(index_a) - point(index_b)))


def find_hairline(mask, start, direction, max_steps):
    """Walk from start along direction until the first hair pixel is hit."""
    h, w = mask.shape
    for step in range(max_steps):
        x = int(round(start[0] + direction[0] * step))
        y = int(round(start[1] + direction[1] * step))
        if not (0 <= x < w and 0 <= y < h):
            break
        if mask[y, x] == 255:
            return np.array([x, y], dtype=float)
    return None


glabella_point = point(9)

# Direction of the face's vertical axis, pointing from chin up toward forehead
face_axis = point(10) - point(152)
face_axis = face_axis / np.linalg.norm(face_axis)

hairline_point = find_hairline(
    hair_mask,
    start=glabella_point,
    direction=face_axis,
    max_steps=int(distance(10, 152) * 0.8),
)


def angle_at_vertex(point_a, vertex, point_b):
    """Return the angle at vertex in degrees."""
    vector_a = point_a - vertex
    vector_b = point_b - vertex

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return None

    cosine = np.dot(vector_a, vector_b) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# Calculate measurements
glabella_point = point(9)  # Approximate point between the eyebrows

forehead_height_px = (
    float(np.linalg.norm(hairline_point - glabella_point))
    if hairline_point is not None
    else None
)

cheekbone_width_px = distance(234, 454)
eye_width_left_px = distance(33, 133)
eye_width_right_px = distance(362, 263)
distance_between_eyes_px = distance(133, 362)
nose_width_px = distance(98, 327)
mouth_width_px = distance(61, 291)
forehead_width_px = distance(54, 284)
face_width_px = distance(93, 323)
jaw_width_px = distance(132, 361)
face_height_px = distance(10, 152)
vertical_face_length_px = (
    float(np.linalg.norm(hairline_point - point(152)))
    if hairline_point is not None
    else None
)
midface_to_chin_px = distance(168, 152)

# This is the angle across the chin, not the jaw corners.
right_jaw_angle_deg = angle_at_vertex(
    point(397),  # jaw corner candidate
    point(172),  # point along the jaw
    point(152),  # chin
)
print(f"Right jaw angle: {right_jaw_angle_deg:.1f} degrees")

# Draw all landmarks in green
output = image.copy()

for landmark in landmarks:
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    cv2.circle(output, (x, y), 2, (0, 255, 0), -1)

for index, landmark in enumerate(landmarks):
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    # Approximate cheek regions on each side of a front-facing face
    in_cheek_region = (
        0.10 <= landmark.x <= 0.22 or 0.78 <= landmark.x <= 0.90
    ) and 0.48 <= landmark.y <= 0.62

    if in_cheek_region:
        cv2.circle(output, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(
            output,
            str(index),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

# Highlight and label landmarks used in measurements
landmark_ids = [
    9,
    10,
    33,
    54,
    61,
    93,
    98,
    133,
    152,
    168,
    172,
    263,
    284,
    291,
    323,
    327,
    362,
    397,
]

for index in landmark_ids:
    x, y = point(index).astype(int)
    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(
        output,
        str(index),
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

if hairline_point is not None:
    hair_x, hair_y = hairline_point.astype(int)
    cv2.circle(output, (hair_x, hair_y), 6, (255, 0, 0), -1)

if hairline_point is not None:
    # Gets the hairline's x and y coordinates and converts them to whole numbers
    hair_x, hair_y = hairline_point.astype(int)
    # Gets the landmarks 9 pixel coordinates adn converts them to integers
    glabella_x, glabella_y = point(9).astype(int)

    # draws a line from the blue dot to 9
    cv2.line(output, (hair_x, hair_y), (glabella_x, glabella_y), (255, 255, 0), 2)

    # Checks if the distance was calculated before trying to display it
    if forehead_height_px is not None:
        # Finds the halfway point between the two coordinates
        midpoint = ((hair_x + glabella_x) // 2, (hair_y + glabella_y) // 2)
        cv2.putText(
            output,
            f"{forehead_height_px:.1f} px",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )


# Resize the display copy to fit on screen while keeping the whole image
display_scale = min(900 / width, 700 / height, 1.0)
display_width = int(width * display_scale)
display_height = int(height * display_scale)

display_image = cv2.resize(output, (display_width, display_height))

cv2.imshow("Face landmarks", display_image)

# Print measurements
measurements = {
    "forehead_height_px": forehead_height_px,
    "eye_width_left_px": eye_width_left_px,
    "eye_width_right_px": eye_width_right_px,
    "distance_between_eyes_px": distance_between_eyes_px,
    "nose_width_px": nose_width_px,
    "mouth_width_px": mouth_width_px,
    "forehead_width_px": forehead_width_px,
    "face_width_px": face_width_px,
    "jaw_width_px": jaw_width_px,
    "face_height_px": face_height_px,
    "vertical_face_length_px": vertical_face_length_px,
    "midface_to_chin_px": midface_to_chin_px,
    "chin_angle_deg": right_jaw_angle_deg,
    "cheekbone_width_px": cheekbone_width_px,
}


cv2.waitKey(0)
cv2.destroyAllWindows()


def cm(pixels):
    return None if pixels is None else pixels * 0.00518


forehead_height_px = measurements["forehead_height_px"]
eye_width_left_px = measurements["eye_width_left_px"]
eye_width_right_px = measurements["eye_width_right_px"]
distance_between_eyes_px = measurements["distance_between_eyes_px"]
nose_width_px = measurements["nose_width_px"]
mouth_width_px = measurements["mouth_width_px"]
forehead_width_px = measurements["forehead_width_px"]
cheekbone_width_px = measurements["cheekbone_width_px"]
face_width_px = measurements["face_width_px"]
jaw_width_px = measurements["jaw_width_px"]
face_height_px = measurements["face_height_px"]
vertical_face_length_px = measurements["vertical_face_length_px"]
midface_to_chin_px = measurements["midface_to_chin_px"]

forehead_height = cm(forehead_height_px)
eye_width_left = cm(eye_width_left_px)
eye_width_right = cm(eye_width_right_px)
distance_between_eyes = cm(distance_between_eyes_px)
nose_width = cm(nose_width_px)
mouth_width = cm(mouth_width_px)
forehead_width = cm(forehead_width_px)
face_width = cm(face_width_px)
jaw_width = cm(jaw_width_px)
face_height = cm(face_height_px)
vertical_face_length = cm(vertical_face_length_px)
midface_to_chin = cm(midface_to_chin_px)
cheekbone_width = cm(cheekbone_width_px)


"""
jaw width: check
jaw_width ratio  = jaw width / cheekbone_width
jaw taper width = chin_width / jaw_width


square: acute angle

"""


def get_jaw_type(jaw_width, chin_width, cheekbone_width, chin_angle, forehead_width):
    """
    chin angle
    relatively low: x < 65
    low: 65 <= x < 85
    moderate: 85 <= x < 105
    relatively high: 105 <= x < 120
    high: x >= 120

    jaw_cheekbone_ratio
    relatively low: x < 0.7
    low: 0.7 <= x < 0.8
    moderate: 0.8 <= x < 0.9
    relatively high: 0.9 <= x < 0.95
    high: x >= 0.95

    jaw_taper
    relatively low: x < 0.25
    low: 0.25 - 0.34
    moderate: 0.35 - 0.44
    relatively high: 0.45 - 0.54
    high: x > 0.54

    forehead_jaw_ratio
    relatively low: x < 0.85
    low: 0.85 <= x < 0.95
    moderate: 0.95 <= x < 1.05
    relatively high: 1.05 <= x < 1.15
    high: x >= 1.15

    """
    forehead_jaw_ratio = forehead_width / jaw_width
    jaw_cheekbone_ratio = jaw_width / cheekbone_width
    jaw_type = ""

    if chin_angle < 65:
        # YES
        if forehead_jaw_ratio >= 1.15:
            jaw_type = "heart"
        else:
            jaw_type = "v"
    else:
        # If the chin angle is moderate
        if chin_angle >= 85 and chin_angle < 105:
            # If the jaw cheekbone ratio is moderate
            if jaw_cheekbone_ratio >= 0.8 and jaw_cheekbone_ratio < 0.9:
                jaw_type = "oval"
            else:
                jaw_type = "angular"
        # If the chin angle is not moderate
        if jaw_cheekbone_ratio < 0.7:
            jaw_type = "round"
        else:
            jaw_type = "square"
    return jaw_type


chin_angle_deg = angle_at_vertex(
    point(172),  # one jaw side
    point(152),  # chin: the vertex
    point(397),  # other jaw side
)


def jaw_features():  # <-- paste the new function here
    # Chin width near the bottom, relative to the jaw corners
    chin_to_jaw = distance(149, 378) / distance(132, 361)

    # Angle at the chin: flat/boxy chin = large angle, curved/pointed = smaller
    chin_angle = angle_at_vertex(point(149), point(152), point(378))

    # Angle at the jaw corner (132): sharp corner = closer to 90, round = larger
    jaw_corner_angle = angle_at_vertex(point(234), point(132), point(152))

    return chin_to_jaw, chin_angle, jaw_corner_angle


def roughly_equal(a, b, tolerance=0.10):
    """Return True if two measurements are within 10% of each other."""
    largest = max(a, b)
    return largest > 0 and abs(a - b) / largest <= tolerance


def is_diamond(f):
    return f["forehead_cheek"] < 0.85 and f["jaw_cheek"] < 0.85


def is_heart(f):
    return f["forehead_jaw"] >= 1.12 and f["jaw_cheek"] < 0.85


def is_triangle(f):
    return f["jaw_forehead"] >= 1.20


def is_oblong(f):
    return f["length"] >= 1.40


def is_square(f):
    return (
        1.18 <= f["length"] < 1.25
        and f["jaw_cheek"] >= 0.90
        and f["jaw_corner"] < 143.5
    )


def is_round(f):
    return f["length"] < 1.18 and f["jaw_cheek"] >= 0.75 and f["jaw_corner"] >= 143.5


def is_oval(f):
    return (
        1.25 <= f["length"] < 1.40
        and f["forehead_cheek"] >= 0.85
        and f["jaw_cheek"] >= 0.85
        and f["jaw_corner"] >= 145
    )


def main():
    # --- Tunable thresholds ---
    DIAMOND_MAX = 0.85  # forehead AND jaw below this fraction of cheekbones -> diamond
    HEART_RATIO = 1.12  # forehead/jaw at or above this -> heart
    TRIANGLE_RATIO = 1.10  # jaw/forehead at or above this -> triangle
    LONG = 1.20  # height/cheekbone width at or above this = long face
    VERY_LONG = 1.40  # long face at or above this = oblong instead of oval
    ANGULAR_TAPER = 0.70  # lower-jaw / jaw-corner width at or above this = angular jaw

    # --- Measurements (all ratios, so units don't matter) ---
    length_ratio = face_height / cheekbone_width
    forehead_to_cheek = forehead_width / cheekbone_width
    jaw_to_cheek = jaw_width / cheekbone_width
    forehead_to_jaw = forehead_width / jaw_width
    jaw_to_forehead = jaw_width / forehead_width
    # Calculate these before the classification rules
    chin_to_jaw, chin_angle, jaw_corner_angle = jaw_features()

    is_long = length_ratio >= LONG
    angular_jaw = jaw_corner_angle < 135

    features = {
        "length": length_ratio,
        "forehead_cheek": forehead_to_cheek,
        "jaw_cheek": jaw_to_cheek,
        "forehead_jaw": forehead_to_jaw,
        "jaw_forehead": jaw_to_forehead,
        "jaw_corner": jaw_corner_angle,
    }

    shape_checks = {
        "diamond": is_diamond,
        "heart": is_heart,
        "triangle": is_triangle,
        "square": is_square,
        "round": is_round,
        "oblong": is_oblong,
        "oval": is_oval,
    }

    matches = [name for name, check in shape_checks.items() if check(features)]

    if len(matches) == 1:
        print(f"Face shape: {matches[0]}")
    elif len(matches) > 1:
        print(f"Ambiguous measurements; matching rules: {', '.join(matches)}")
    else:
        print("No shape rule matched; review the measurements.")


if __name__ == "__main__":
    main()
