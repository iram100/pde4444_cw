import cv2
import time
from collections import deque
from predict import predict

cap = cv2.VideoCapture(0)

history = deque(maxlen=5)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ---- CENTER BOX (slightly bigger but not restrictive) ----
    x1, y1 = int(w * 0.25), int(h * 0.25)
    x2, y2 = int(w * 0.75), int(h * 0.75)

    roi = frame[y1:y2, x1:x2]

    label, confidence = predict(roi)

    # ---- SMOOTHING ----
    history.append(label)
    final_label = max(set(history), key=history.count)

    # ---- COLORS ----
    if final_label == "PASS":
        color = (0, 200, 0)
    else:
        color = (0, 0, 255)

    # ---- DRAW BOX ----
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # ---- BORDER ----
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), color, 3)

    # ---- TEXT BG ----
    cv2.rectangle(frame, (10, 10), (350, 80), color, -1)

    # ---- TEXT ----
    text = f"{final_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)

    # ---- INSTRUCTION ----
    cv2.putText(frame, "Place bottle inside box",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    # ---- FPS ----
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    cv2.imshow("Bottle Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()