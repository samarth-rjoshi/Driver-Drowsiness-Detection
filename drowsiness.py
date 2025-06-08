from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import time

# Constants
FRAME_SAVE_PATH = "frames"  # Directory to save frames
EAR_THRESH = 0.3  # Eye Aspect Ratio threshold to indicate blink
EAR_CONSEC_FRAMES = 30  # Number of consecutive frames for drowsiness detection
SHAPE_PREDICTOR_PATH = ""  # Path to the shape predictor file

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize counters and flags
count = 0
total_frames = 0
drowsy_frames = 0
running = True

# Create directory to save frames
if not os.path.exists(FRAME_SAVE_PATH):
    os.makedirs(FRAME_SAVE_PATH)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Get the indices of the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start video capture
cam = cv2.VideoCapture(0)

while running:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # Predict facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw contours around the eyes
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)

        # Check if EAR is below the threshold
        if ear < EAR_THRESH:
            count += 1

            # If eyes are closed for a sufficient number of frames, alert drowsiness
            if count >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                drowsy_frames += 1
        else:
            count = 0

    # Update total frames and calculate accuracy
    total_frames += 1
    accuracy = (1 - drowsy_frames / total_frames) * 100

    # Display accuracy on the frame
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the frame
    frame_filename = f"frame_{total_frames:04d}.jpg"
    cv2.imwrite(os.path.join(FRAME_SAVE_PATH, frame_filename), frame)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()