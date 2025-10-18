import cv2
import numpy as np
import os

class VideoBallDetector:
    def __init__(self, video_path):
        """Initialize video ball detector"""
        print(f" Loading video: {video_path}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f" Video file not found: {video_path}")
            exit(1)
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f" Could not open video file: {video_path}")
            exit(1)
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Ball detection settings
        self.orange_color = [0, 140, 255]  # BGR format
        self.detection_count = 0
        self.ball_tracks = []  # Store all ball positions
        
        # Video output settings (optional)
        self.save_output = False
        self.output_writer = None
        
        print(" Video loaded successfully!")
        print(f"Video Info:")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.1f}")
        print(f"   Total Frames: {self.total_frames}")
        print(f"   Duration: {self.duration:.1f} seconds")
        
    def get_video_limits(self, color_bgr):
        """Enhanced HSV limits for precise ball detection"""
        c = np.uint8([[color_bgr]])
        hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        hue = hsvC[0][0][0]
        
        # Precise range for ball detection
        lowerLimit = np.array([hue - 10, 120, 120], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
        return lowerLimit, upperLimit

    def detect_ball_in_frame(self, frame):
        """
        Detect ball in a single frame
        Returns ball info or None
        """
        # Convert to HSV for color detection
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create color mask
        lower_limit, upper_limit = self.get_video_limits(self.orange_color)
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, mask
        
        # Find best ball contour
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filter (adjust for your video)
            if area < 25 or area > 50000:  # Wider range for video
                continue
            
            # Circularity test
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity < 0.5:  # More lenient for video
                continue
            
            score = circularity * area
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None, mask
        
        # Get ball center and radius
        (center_x, center_y), radius = cv2.minEnclosingCircle(best_contour)
        center_x, center_y, radius = int(center_x), int(center_y), int(radius)
        
        return {
            'center': (center_x, center_y),
            'radius': radius,
            'contour': best_contour,
            'area': cv2.contourArea(best_contour),
            'circularity': 4 * np.pi * cv2.contourArea(best_contour) / (cv2.arcLength(best_contour, True) ** 2)
        }, mask

    def draw_ball_with_trail(self, frame, ball_info, frame_number, timestamp):
        """Draw ball detection with trajectory trail"""
        if ball_info is None:
            return frame
        
        center_x, center_y = ball_info['center']
        radius = ball_info['radius']
        
        # Store ball position for trail
        self.ball_tracks.append((center_x, center_y, frame_number, timestamp))
        
        # Keep only recent positions (trail length)
        max_trail_length = 30
        if len(self.ball_tracks) > max_trail_length:
            self.ball_tracks.pop(0)
        
        # Draw trail (previous positions)
        for i in range(len(self.ball_tracks) - 1):
            if i < len(self.ball_tracks) - 1:
                pt1 = (self.ball_tracks[i][0], self.ball_tracks[i][1])
                pt2 = (self.ball_tracks[i + 1][0], self.ball_tracks[i + 1][1])
                
                # Fade trail color based on age
                trail_intensity = int(255 * (i + 1) / len(self.ball_tracks))
                cv2.line(frame, pt1, pt2, (0, trail_intensity, trail_intensity), 2)
        
        # Draw precise circle around current ball
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 3)
        cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)  # Center dot
        
        # Draw crosshairs
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 0, 0), 2)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 0, 0), 2)
        
        # Enhanced information display
        info_y = center_y - radius - 60
        
        cv2.putText(frame, f"BALL #{self.detection_count}", 
                    (center_x - radius, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Position: ({center_x}, {center_y})", 
                    (center_x - radius, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Size: {radius*2}px", 
                    (center_x - radius, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Time: {timestamp:.2f}s", 
                    (center_x - radius, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame

    def setup_output_video(self, output_path):
        """Setup video output writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        self.save_output = True
        print(f" Output video will be saved to: {output_path}")

    def process_video(self, save_output_path=None, skip_frames=1):
        """
        Process entire video for ball detection
        skip_frames: Process every N frames (1=all, 2=every other, etc.)
        """
        if save_output_path:
            self.setup_output_video(save_output_path)
        
        print(" Starting video ball detection...")
        print(" Controls:")
        print("   'q' - Quit")
        print("   'SPACE' - Pause/Resume")
        print("   's' - Save current frame")
        print("   'f' - Skip to next frame (when paused)")
        print("   'r' - Restart video from beginning")
        
        frame_number = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    # Read frame from video
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print(" End of video reached")
                        break
                    
                    frame_number += 1
                    
                    # Skip frames if needed (for performance)
                    if frame_number % skip_frames != 0:
                        continue
                    
                    # Calculate timestamp
                    timestamp = frame_number / self.fps
                    
                    # Detect ball in current frame
                    ball_info, mask = self.detect_ball_in_frame(frame)
                    
                    if ball_info is not None:
                        self.detection_count += 1
                        frame = self.draw_ball_with_trail(frame, ball_info, frame_number, timestamp)
                        
                        # Print detection info
                        center = ball_info['center']
                        print(f" Frame {frame_number}: Ball at ({center[0]}, {center[1]}) - Time: {timestamp:.2f}s")
                    
                    else:
                        cv2.putText(frame, "No orange ball detected", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add video info overlay
                progress = (frame_number / self.total_frames) * 100
                cv2.putText(frame, f"Video Ball Detection - Frame {frame_number}/{self.total_frames} ({progress:.1f}%)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Detections: {self.detection_count} | Time: {timestamp:.1f}s/{self.duration:.1f}s", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                status = "PAUSED" if paused else "PLAYING"
                cv2.putText(frame, status, (self.width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 165, 255) if paused else (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow(' Video Ball Detection', frame)
                cv2.imshow('Color Mask (Debug)', mask)
                
                # Save frame to output video
                if self.save_output and self.output_writer:
                    self.output_writer.write(frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  
                    paused = not paused
                    print(f"{' PAUSED' if paused else '  PLAYING'}")
                elif key == ord('s'):
                    filename = f'video_ball_frame_{frame_number}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")
                elif key == ord('f') and paused:
                    # Skip one frame when paused
                    ret, frame = self.cap.read()
                    if ret:
                        frame_number += 1
                        print(f" Skipped to frame {frame_number}")
                elif key == ord('r'):
                    # Restart video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_number = 0
                    self.detection_count = 0
                    self.ball_tracks.clear()
                    print(" Video restarted")
        
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            if self.output_writer:
                self.output_writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n Video processing complete!")
            print(f" Final Results:")
            print(f"   Total frames processed: {frame_number}")
            print(f"   Ball detections: {self.detection_count}")
            print(f"   Detection rate: {(self.detection_count/frame_number)*100:.1f}%")
            if self.save_output:
                print(f"   Output video saved successfully!")

def select_video_file():
    """Simple video file selector"""
    print(" Video Ball Detection System")
    print("=" * 40)
    
    # Common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    # Look for video files in current directory
    current_dir = os.getcwd()
    video_files = []
    
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if video_files:
        print(f" Found {len(video_files)} video file(s) in current directory:")
        for i, video in enumerate(video_files):
            print(f"   {i + 1}. {video}")
        
        print(f"\n   {len(video_files) + 1}. Enter custom path")
        
        try:
            choice = int(input(f"\nSelect video (1-{len(video_files) + 1}): "))
            
            if 1 <= choice <= len(video_files):
                return video_files[choice - 1]
            elif choice == len(video_files) + 1:
                return input("Enter video file path: ")
            else:
                print("Invalid choice")
                return None
        except ValueError:
            print("Invalid input")
            return None
    else:
        print(" No video files found in current directory")
        return input("Enter video file path: ")

def main():
    # Select video file
    video_path = select_video_file()
    
    if not video_path:
        print(" No video selected")
        return
    
    # Create detector
    detector = VideoBallDetector(video_path)
    
    # Ask about output video
    save_output = input("\n Save output video with detections? (y/n): ").lower() == 'y'
    output_path = None
    
    if save_output:
        output_path = f"ball_detection_output_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
        print(f"Output will be saved as: {output_path}")
    
    # Process video
    detector.process_video(save_output_path=output_path)

if __name__ == "__main__":
    main()
