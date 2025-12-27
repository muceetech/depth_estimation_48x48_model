import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================
# CONFIG
# =========================================================
ESP32_STREAM_URL = "http://10.98.95.1:81/stream"  # CHANGE IP
MODEL_PATH = "pydnet48.pth"
IMG_SIZE = 48
DEVICE = torch.device("cpu")

# =========================================================
# EXACT PyDNet48 MODEL (MATCHES YOUR TRAINING CODE)
# =========================================================
class PyDNet48(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)  # 24×24

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # 12×12

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )

        # Depth prediction (12×12)
        self.pred3 = nn.Conv2d(128, 1, 1)

        # Refinement to 24×24
        self.refine2 = nn.Sequential(
            nn.Conv2d(64 + 1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pred2 = nn.Conv2d(64, 1, 1)

        # Refinement to 48×48
        self.refine1 = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pred1 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        f1 = self.conv1(x)                 # 48×48
        f2 = self.conv2(self.pool1(f1))    # 24×24
        f3 = self.conv3(self.pool2(f2))    # 12×12

        d3 = torch.sigmoid(self.pred3(f3))

        d3_up = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        r2 = self.refine2(torch.cat([f2, d3_up], dim=1))
        d2 = torch.sigmoid(self.pred2(r2))

        d2_up = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        r1 = self.refine1(torch.cat([f1, d2_up], dim=1))
        d1 = torch.sigmoid(self.pred1(r1))

        return d1, d2, d3

# =========================================================
# LOAD MODEL
# =========================================================
model = PyDNet48().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

print("✅ PyDNet48 model loaded successfully")

# =========================================================
# OPEN ESP32-CAM STREAM
# =========================================================
cap = cv2.VideoCapture(ESP32_STREAM_URL)
if not cap.isOpened():
    print("❌ Cannot open ESP32-CAM stream")
    exit()

print("✅ ESP32-CAM stream connected")

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed")
        break

    # Rotate camera 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # -----------------------------
    # PREPROCESS (MUST MATCH TRAINING)
    # -----------------------------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))      # CHW
    input_tensor = torch.tensor(img).unsqueeze(0).to(DEVICE)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    with torch.no_grad():
        depth48, depth24, depth12 = model(input_tensor)

    depth = depth48.squeeze().cpu().numpy()

    # -----------------------------
    # DISPLAY DEPTH
    # -----------------------------
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imshow("ESP32-CAM", frame)
    cv2.imshow("Depth (48x48)", cv2.resize(depth_color, (240, 240)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# CLEANUP
# =========================================================
cap.release()
cv2.destroyAllWindows()
