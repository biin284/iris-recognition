import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os

# --- CẤU HÌNH HỆ THỐNG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_PATH = 'iris_unet_v3.pth'
ARCFACE_PATH = 'iris_arcface_best.pth'
REF_IMAGES_DIR = 'my_data/Long'
THRESHOLD = 0.50  # Nâng lên 0.50 để đảm bảo an toàn tuyệt đối
MIN_IRIS_RATIO = 0.035  # Chấp nhận ảnh của bạn ở mức 4-5% diện tích
MIN_CIRCULARITY = 0.65  # Bộ lọc quan trọng: Loại bỏ tóc/trán (mống mắt ~ 0.8-1.0)

arc_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class IrisArcFace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(960, 512),
            torch.nn.BatchNorm1d(512)
        )

    def forward(self, x): return self.backbone(x)


def load_models():
    unet = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=1, activation=None).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet.eval()
    arcface = IrisArcFace().to(DEVICE)
    sd = torch.load(ARCFACE_PATH, map_location=DEVICE)
    arcface.load_state_dict({k: v for k, v in sd.items() if 'head' not in k}, strict=False)
    arcface.eval()
    return unet, arcface


def extract_embedding_v4(frame, unet, arcface):
    h, w = frame.shape[:2]
    roi_area = h * w

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (256, 256))
    img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_input).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask_logits = unet(img_tensor)
        mask = (torch.sigmoid(mask_logits).cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, 0, "NO TARGET"

    # Tìm mống mắt tốt nhất bằng cách kiểm tra độ tròn
    best_cnt = None
    max_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < roi_area * 0.005: continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue

        # Công thức tính độ tròn (Circularity)
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        if circularity > MIN_CIRCULARITY:
            if area > max_score:
                max_score = area
                best_cnt = cnt

    if best_cnt is None: return None, 0, "NOT AN EYE (SHAPE ERR)"

    actual_ratio = cv2.contourArea(best_cnt) / roi_area
    if actual_ratio < MIN_IRIS_RATIO: return None, actual_ratio, f"SMALL ({actual_ratio:.1%})"

    # Xử lý ảnh mống mắt sạch
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    tensor = arc_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = F.normalize(arcface(tensor), p=2, dim=1)

    return embedding, actual_ratio, "OK"


# --- MAIN ---
unet, arcface = load_models()

print("⌛ Đang học mẫu từ TẤT CẢ các ảnh bạn chụp...")
embeddings = []
# Sử dụng toàn bộ kho ảnh mẫu Long1, Long2...
for f in os.listdir(REF_IMAGES_DIR):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(REF_IMAGES_DIR, f))
        if img is not None:
            # Khi học mẫu, ta lấy đặc trưng sạch nhất
            emb, _, status = extract_embedding_v4(img, unet, arcface)
            if emb is not None: embeddings.append(emb)

if not embeddings:
    print("❌ Lỗi: Không thể học được gì từ ảnh mẫu. Hãy kiểm tra lại ánh sáng ảnh mẫu.")
    exit()

ref_embedding = torch.mean(torch.stack(embeddings), dim=0)
ref_embedding = F.normalize(ref_embedding, p=2, dim=1)
print(f"✅ Đã đăng ký thành công danh tính với {len(embeddings)} mẫu đặc trưng.")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    roi_size = 260
    x1, y1 = w // 2 - roi_size // 2, h // 2 - roi_size // 2
    x2, y2 = x1 + roi_size, y1 + roi_size

    roi = frame[y1:y2, x1:x2]
    curr_emb, ratio, status = extract_embedding_v4(roi, unet, arcface)

    if curr_emb is None:
        cv2.putText(frame, f"STATUS: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        roi_color = (0, 165, 255)
    else:
        score = torch.mm(curr_emb, ref_embedding.t()).item()
        color = (0, 255, 0) if score > THRESHOLD else (0, 0, 255)
        text = f"SCORE: {score:.4f}"
        if score > THRESHOLD: cv2.putText(frame, "ACCESS GRANTED", (30, 80), 1, 1, color, 2)
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        roi_color = color

    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
    cv2.imshow("Iris Guard v4 - FINAL", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()