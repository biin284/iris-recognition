import os
import cv2
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from collections import defaultdict

# --- CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_PATH = 'iris_unet_v3.pth'
ARCFACE_PATH = 'iris_arcface_best.pth'
UBIRIS_PATH = r'E:\iris_reconigtion\CLASSES_400_300_Part2'
THRESHOLD_KAGGLE = 0.222


# [Định nghĩa lớp IrisArcFace và hàm load_system, process_image giữ nguyên như cũ]
class IrisArcFace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(960, 512), torch.nn.BatchNorm1d(512)
        )

    def forward(self, x): return self.backbone(x)


def load_system():
    unet = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=1, activation=None).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    arcface = IrisArcFace().to(DEVICE)
    sd = torch.load(ARCFACE_PATH, map_location=DEVICE)
    arcface.load_state_dict({k: v for k, v in sd.items() if 'head' not in k}, strict=False)
    unet.eval();
    arcface.eval()
    return unet, arcface


def process_image(path, unet, arcface):
    start_time = time.time()
    raw_img = cv2.imread(path)
    if raw_img is None: return None, 0
    h, w = raw_img.shape[:2]
    img_input = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (256, 256))
    img_tensor = torch.from_numpy(img_input.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = (torch.sigmoid(unet(img_tensor)).cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h))
    masked = cv2.bitwise_and(raw_img, raw_img, mask=mask)
    l, a, b = cv2.split(cv2.cvtColor(masked, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = transform(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(arcface(tensor), p=2, dim=1)
    return emb, time.time() - start_time


# --- CHƯƠNG TRÌNH TEST CHO CẤU TRÚC ẢNH PHẲNG ---
unet, arcface = load_system()

# Bước 1: Gom nhóm ảnh theo ID (Dựa vào ký tự trước dấu gạch dưới đầu tiên)
id_map = defaultdict(list)
for f in os.listdir(UBIRIS_PATH):
    if f.lower().endswith(('.jpg', '.png', '.tiff', '.jpeg')):
        # Tách lấy ID (ví dụ: C252_1.jpg -> C252)
        target_id = f.split('_')[0]
        id_map[target_id].append(os.path.join(UBIRIS_PATH, f))

# Lọc các ID có ít nhất 2 ảnh để test Genuine
valid_ids = [k for k, v in id_map.items() if len(v) >= 2]

if len(valid_ids) < 2:
    print("❌ Lỗi: Không đủ dữ liệu ID để thực hiện so sánh chéo.")
    exit()

test_ids = random.sample(valid_ids, min(100, len(valid_ids)))

print(f"✅ Đang test {len(test_ids)} danh tính từ cấu trúc ảnh phẳng...")
print(f"{'Identity':<15} | {'Type':<10} | {'Score':<10} | {'Result':<10} | {'Latency (s)'}")
print("-" * 75)

latencies = []
for target_id in test_ids:
    imgs = id_map[target_id]

    # Genuine Test
    emb1, t1 = process_image(imgs[0], unet, arcface)
    emb2, t2 = process_image(imgs[1], unet, arcface)
    score_gen = torch.mm(emb1, emb2.t()).item()
    res_gen = "PASS" if score_gen > THRESHOLD_KAGGLE else "FAIL"
    print(f"{target_id:<15} | {'Genuine':<10} | {score_gen:.4f} | {res_gen:<10} | {t1 + t2:.3f}")

    # Imposter Test
    other_id = random.choice([i for i in valid_ids if i != target_id])
    other_img = id_map[other_id][0]
    emb_other, t3 = process_image(other_img, unet, arcface)
    score_imp = torch.mm(emb1, emb_other.t()).item()
    res_imp = "PASS" if score_imp < THRESHOLD_KAGGLE else "FAIL (FA)"
    print(f"{'vs ' + other_id:<15} | {'Imposter':<10} | {score_imp:.4f} | {res_imp:<10} | {t1 + t3:.3f}")

    latencies.extend([t1, t2, t3])

if latencies:
    print("-" * 75)
    print(f"✅ Latency trung bình: {np.mean(latencies):.3f} giây/ảnh")