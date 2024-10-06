import cv2
import dlib
from scipy.spatial import distance

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# Thresholds and frame count constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 15

# Initialize counters
COUNTER = 0
YAWN_COUNTER = 0

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Eye and mouth landmark indices
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
(mStart, mEnd) = (48, 68)

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read a frame from the webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # Condition 1: Face Detection (Detected or Not Detected)
    if len(rects) == 0:
        print("No faces detected.")
    else:
        print(f"Number of faces detected: {len(rects)}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Draw eye and mouth contours for debugging
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        for (x, y) in mouth:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        # Condition 2: Eyes Normal Blinking, Half-Closed, and Fully Closed
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            print(f"Eyes are closing. EAR: {ear:.2f} Counter: {COUNTER}")
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                print("Drowsiness detected!")
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if COUNTER > 0:
                print("Normal blink or half-closed detected.")
            COUNTER = 0

        # Condition 3: Mouth Normal Opening, Half-Opening, and Yawning
        if mar > MOUTH_AR_THRESH:
            YAWN_COUNTER += 1
            print(f"Mouth is opening. MAR: {mar:.2f} Yawn Counter: {YAWN_COUNTER}")
            if YAWN_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                print("Yawning detected!")
                cv2.putText(frame, "YAWNING DETECTED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            if YAWN_COUNTER > 0:
                print("Normal or half mouth opening detected.")
            YAWN_COUNTER = 0

    # Show the frame with detections
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()