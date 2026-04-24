import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class BottleCNN(nn.Module):
    def __init__(self):
        super(BottleCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),  
            nn.ReLU(),
            nn.Dropout(0.5),               
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ---- LOAD MODEL ----
model = BottleCNN()
model.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))
model.eval()


# ---- PREPROCESS ----
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))

    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32)
    return img


def predict(frame, debug=True):
    img = preprocess(frame)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    # Convert model prediction
    label = "PASS" if pred.item() == 1 else "FAIL"
    confidence = confidence.item()

    

    return label, confidence

