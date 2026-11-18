import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the photo
image = cv2.imread("shape/faces.jpeg")

if image is None:
    raise FileNotFoundError("Could not find shape/faces.jpeg")

# Convert the photo from BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Prepare the image for MediaPipe
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# Load the hair segmentation model
base_options = python.BaseOptions(model_asset_path="shape/hair_segmenter.tflite")

options = vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_category_mask=True,
    output_confidence_masks=False,
)

# Run the segmentation model
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    result = segmenter.segment(mp_image)
    mask = result.category_mask.numpy_view().copy()

# Create a binary hair mask
hair_mask = (mask == 1).astype(np.uint8) * 255

# Get the image dimensions
height, width = image.shape[:2]

# Resize the mask if needed
if hair_mask.shape != (height, width):
    hair_mask = cv2.resize(hair_mask, (width, height), interpolation=cv2.INTER_NEAREST)

# Find the center column
center_x = width // 2

# Define the vertical search area
top_y = int(height * 0.10)
bottom_y = int(height * 0.45)

# Get hair pixels in the center column
column = hair_mask[top_y:bottom_y, center_x]
hair_pixels = np.where(column == 255)[0]

if len(hair_pixels) > 0:
    # Find the lowest hair pixel in the search area
    hairline_y = top_y + hair_pixels.max()

    # Save the estimated hairline point
    hairline_point = (center_x, hairline_y)

    # Draw a red dot
    cv2.circle(image, hairline_point, 7, (0, 0, 255), -1)

    print("Estimated center hairline:", hairline_point)

else:
    print("No hair detected at the center. Try adjusting the search area.")

# Display the result
cv2.imshow("Center Hairline Point", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
