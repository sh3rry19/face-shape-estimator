import cv2
import dlib
import math


# measurements
eye_width = 0
face_height = 0
face_width = 0
eye_distance = 0
nose_width = 0
nose_height = 0
mouth_width = 0
forehead_width = 0
cheekbone_width = 0
jaw_width = 0

face_shapes = [
    "rectangle",
    "oblong",
    "pear",
    "heart",
    "oval",
    "diamond",
    "traingle",
    "square",
]

image = cv2.imread("shape/faces.jpeg")

if image is None:
    raise FileNotFoundError("Could not find shape/faces.jpeg")

height, width = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape/shape_predictor_68_face_landmarks.dat")

faces = detector(gray)
print(f"Faces detected {len(faces)}")

for face in faces:
    landmarks = predictor(gray, face)

    for i in range(68):
        point = landmarks.part(i)

        x = point.x
        y = point.y

        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        cv2.putText(
            image,
            str(i),
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )


cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
