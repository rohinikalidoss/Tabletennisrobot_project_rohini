import cv2
from PIL import Image
import numpy as np

def find_available_cameras(max_cameras=10):
    """
    Detect all available cameras (built-in + external)
    Returns list of working camera indices
    """
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Get camera info
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"📷 Camera {i}: {int(width)}x{int(height)} @ {fps:.1f}FPS")
                available_cameras.append(i)
            cap.release()
        
    return available_cameras

def get_limits(color_bgr):
    """Enhanced HSV limits for precise ball detection"""
    c = np.uint8([[color_bgr]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]
    
    # More precise range for ball-only detection
    lowerLimit = np.array([hue - 10, 120, 120], dtype=np.uint8)  # Tighter range
    upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    return lowerLimit, upperLimit

def detect_precise_ball(mask):
    """
    Detect ball precisely using contour analysis
    Returns the best ball contour or None
    """
    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    best_contour = None
    best_score = 0
    
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Filter by size (adjust based on expected ball size in your video)
        if area < 200 or area > 20000:  # Minimum and maximum ball area
            continue
        
        # Calculate circularity (how round is the shape)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Ball should be quite round (close to 1.0)
        if circularity < 0.6:  # Adjust this threshold
            continue
        
        # Score this contour (higher is better)
        score = circularity * area
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    return best_contour

def get_tight_bounding_box(contour):
    """
    Get tight bounding box around the ball contour
    Returns (x, y, width, height) that fits exactly around the ball
    """
    # Get the exact bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Get the minimum enclosing circle for comparison
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    
    # Use the circle-based approach for more precise bounds
    radius = int(radius)
    center_x, center_y = int(center_x), int(center_y)
    
    # Create tight square around the circle
    x = center_x - radius
    y = center_y - radius
    w = h = radius * 2
    
    return x, y, w, h, center_x, center_y, radius

# --- DETECT AVAILABLE CAMERAS ---
print(" Detecting available cameras...")
cameras = find_available_cameras()

if not cameras:
    print(" No cameras found!")
    exit()

print(f"\n Found cameras: {cameras}")
print("Camera options:")
for i, cam_id in enumerate(cameras):
    print(f"  {cam_id}: {'Built-in camera' if cam_id == 0 else f'External camera {cam_id}'}")

# --- SELECT CAMERA ---
if len(cameras) == 1:
    selected_camera = cameras[0]
    print(f"\n Auto-selected camera {selected_camera}")
else:
    print(f"\nWhich camera to use? ({'/'.join(map(str, cameras))})")
    try:
        selected_camera = int(input("Enter camera number: "))
        if selected_camera not in cameras:
            print(f"Invalid choice. Using camera {cameras[0]}")
            selected_camera = cameras[0]
    except:
        selected_camera = cameras[0]
        print(f"Using default camera {selected_camera}")

# --- ENHANCED CAMERA SETUP ---
print(f"\n🚀 Starting precise ball detection with camera {selected_camera}...")

# Try different backends for better external camera support
backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
cap = None

for backend in backends:
    try:
        cap = cv2.VideoCapture(selected_camera, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f" Successfully connected using backend")
                break
        cap.release()
    except:
        continue

if cap is None or not cap.isOpened():
    print(f" Could not connect to camera {selected_camera}")
    exit()

# --- OPTIMIZE CAMERA SETTINGS ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Enhanced color for better detection
orange = [0, 140, 255]  # BGR format for orange ball

print(" Tracking orange ball with PRECISE detection...")
print(" Controls: 'q' to quit, 'c' to change camera, 's' to save frame")
print(" The square will fit EXACTLY around the ball!")

frame_count = 0
detection_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    
    # Convert to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get color limits
    lowerLimit, upperLimit = get_limits(color_bgr=orange)
    
    # Create mask
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    
    # Advanced noise reduction for cleaner detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    
    # Detect ball precisely using contour analysis
    ball_contour = detect_precise_ball(mask)
    
    if ball_contour is not None:
        detection_count += 1
        
        # Get tight bounding box that fits exactly around the ball
        x, y, w, h, center_x, center_y, radius = get_tight_bounding_box(ball_contour)
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(x, frame_width - w))
        y = max(0, min(y, frame_height - h))
        
        # Draw PRECISE square around ball (not huge bounding box!)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw center point and circle outline for reference
        cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)  # Center dot
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), 2)  # Circle outline
        
        # Add precise information
        cv2.putText(frame, f"BALL #{detection_count}", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Size: {w}x{h}px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Center: ({center_x},{center_y})", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        # No ball detected
        cv2.putText(frame, "No orange ball detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add camera info overlay
    cv2.putText(frame, f"Camera {selected_camera} - Frame {frame_count} - Detections: {detection_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display both original and mask for debugging
    cv2.imshow(f' PRECISE Ball Detection - Camera {selected_camera}', frame)
    cv2.imshow('Color Mask (Debug)', mask)
    
    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save current frame with detection
        filename = f'precise_ball_detection_{frame_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f" Saved {filename}")
    elif key == ord('c'):
        # Change camera
        print(f"\n Available cameras: {cameras}")
        try:
            new_cam = int(input("Enter new camera number: "))
            if new_cam in cameras:
                cap.release()
                cap = cv2.VideoCapture(new_cam, cv2.CAP_DSHOW)
                if cap.isOpened():
                    selected_camera = new_cam
                    print(f" Switched to camera {new_cam}")
                else:
                    print(f" Could not switch to camera {new_cam}")
        except:
            print("Invalid input")

cap.release()
cv2.destroyAllWindows()
print(f" Detection stopped - Total ball detections: {detection_count}")
