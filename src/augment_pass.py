import cv2
import os
import numpy as np
import random
input_folder = "dataset/raw/PASS"
output_folder = "dataset/raw/PASS_aug"

os.makedirs(output_folder, exist_ok=True)

def augment(img):
    # brightness
    value = random.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=1, beta=value)

    # contrast
    alpha = random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    # slight zoom
    h, w = img.shape[:2]
    scale = random.uniform(0.9, 1.1)
    resized = cv2.resize(img, None, fx=scale, fy=scale)

    # crop center
    startx = resized.shape[1]//2 - w//2
    starty = resized.shape[0]//2 - h//2
    img = resized[starty:starty+h, startx:startx+w]

    return img


count = 0

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    for i in range(1):  # 1 augmentation per image → 300 → 600 total
        aug = augment(img)
        save_path = os.path.join(output_folder, f"aug_{count}.jpg")
        cv2.imwrite(save_path, aug)
        count += 1

print("Augmented images created:", count)