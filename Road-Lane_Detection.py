import cv2
import numpy as np

def detect_lanes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(50, frame.shape[0]), (frame.shape[1]//2 - 45, frame.shape[0]//2 + 60), 
                              (frame.shape[1]//2 + 45, frame.shape[0]//2 + 60), (frame.shape[1] - 50, frame.shape[0])]], 
                            dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough transform
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=4)
    
    # Draw lane lines
    lane_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combine lane image with original frame
    result = cv2.addWeighted(frame, 0.8, lane_image, 1, 0)
    return result

# Open video file or capture from camera
video_capture = cv2.VideoCapture('vid.mp4')

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    processed_frame = detect_lanes(frame)
    cv2.imshow('Lane Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
