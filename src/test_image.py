import cv2
from predict import predict
import os

IMAGE_PATH = "test_images/fail2.jpg"


# ---- LOAD IMAGE ----
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(" Error: Image not found")
    exit()

# ---- PREDICT ----
label, confidence = predict(img)

print(f"Prediction: {label} ({confidence*100:.2f}%)")

# ---- COLOR ----
if label == "PASS":
    color = (0, 200, 0)
else:
    color = (0, 0, 255)

# ---- DRAW RESULT ----
cv2.rectangle(img, (10, 10), (350, 80), color, -1)

cv2.putText(img,
            f"{label} ({confidence*100:.1f}%)",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2)

# ---- SHOW IMAGE ----
cv2.imshow("Test Image Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()