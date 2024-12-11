import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
cap = cv2.VideoCapture(0)
def is_tree_pose(landmarks):
    """Check alignment for Tree Pose."""
    try:
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        
        if abs(left_knee.y - left_ankle.y) < 0.1 and abs(left_hip.y - left_knee.y) < 0.1:
            return True
    except:
        pass
    return False
def is_warrior_1_pose(landmarks):
    """Check alignment for Warrior I Pose."""
    try:
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        shoulders = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        # Warrior I Pose rough alignment checks
        if abs(left_knee.y - left_ankle.y) < 0.2 and abs(right_knee.y - right_ankle.y) < 0.2 and abs(shoulders[0] - shoulders[1]) < 0.1:
            return True
    except:
        pass
    return False


def is_mountain_pose(landmarks):
    """Check alignment for Mountain Pose."""
    try:
        shoulders = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        hips = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

        if abs(shoulders[0] - shoulders[1]) < 0.1 and abs(hips[0] - hips[1]) < 0.1:
            return True
    except:
        pass
    return False


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
        if is_tree_pose(results.pose_landmarks.landmark):
            cv2.putText(
                frame,
                "✅ Tree Pose Detected!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        elif is_warrior_1_pose(results.pose_landmarks.landmark):
            cv2.putText(
                frame,
                "✅ Warrior I Pose Detected!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        elif is_mountain_pose(results.pose_landmarks.landmark):
            cv2.putText(
                frame,
                "✅ Mountain Pose Detected!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "❌ Adjust alignment for poses.",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
    cv2.imshow("Yoga Pose Detection & Feedback", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
