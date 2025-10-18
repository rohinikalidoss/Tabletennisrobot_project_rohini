import cv2
from PIL import Image
import numpy as np

# Function to get HSV color limits for orange
def get_limits(color_bgr):
    """
    Returns lower and upper HSV limits for a given BGR color.
    Used for color detection masks.
    """
    c = np.uint8([[color_bgr]])  # Convert to uint8 type for cvtColor
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]

    # Define a range around the hue
    lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    return lowerLimit, upperLimit

# --- ORANGE BALL COLOR (BGR) ---
# A typical ping-pong ball is bright orange. Let's define that.
orange = [0, 140, 255]  # BGR format (Blue=0, Green=140, Red=255)

# Open the camera
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if multiple cameras

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

print("🎯 Tracking color: ORANGE (BGR = [0, 140, 255])")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get color limits for orange
    lowerLimit, upperLimit = get_limits(color_bgr=orange)

    # Create a mask for the orange color
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Convert to PIL image for bounding box extraction
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Draw rectangle around detected ball
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "Orange Ball Detected", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show both video and mask
    cv2.imshow('Detected Ball', frame)
    cv2.imshow('Color Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()