from ultralytics import YOLO
import cv2

# ---- CONFIG ----
MODEL_PATH = "best.pt"        # your YOLO model
VIDEO_PATH = "test.mp4"       # your video file
CONF = 0.25                    # confidence threshold
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

# 👉 Create resizable window
cv2.namedWindow("YOLO Test", cv2.WINDOW_NORMAL)

# 👉 Set window size (width, height)
cv2.resizeWindow("YOLO Test", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLO Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()