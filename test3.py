import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# --- CẤU HÌNH THÔNG SỐ ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_PATH = 'iris_unet_v3.pth'
ARCFACE_PATH = 'iris_arcface_best.pth'
MY_EYES_DIR = 'my_data/Long'
THRESHOLD = 0.47


# --- 1. LOAD MODEL U-NET (SEGMENTATION) ---
def load_unet():
    model = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=1, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    model.eval()
    return model


# --- 2. LOAD MODEL ARCFACE (RECOGNITION) ---
class IrisArcFace(torch.nn.Module):
    def __init__(self, num_classes=412):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(960, 512),
            torch.nn.BatchNorm1d(512)
        )

    def forward(self, x):
        return self.backbone(x)


def load_arcface():
    model = IrisArcFace().to(DEVICE)
    # Chỉ load phần backbone vì chúng ta không cần lớp phân loại (head) khi nhận diện
    state_dict = torch.load(ARCFACE_PATH, map_location=DEVICE)
    # Lọc bỏ các trọng số của 'head' nếu có
    state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# --- 3. HÀM TRÍCH XUẤT ĐẶC TRƯNG TỔNG HỢP ---
def get_iris_embedding(raw_img_path, unet, arcface):
    raw_img = cv2.imread(raw_img_path)
    h, w = raw_img.shape[:2]

    # BƯỚC A: TẠO MASK BẰNG U-NET
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (256, 256))
    img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_input).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask_logits = unet(img_tensor)
        mask = (torch.sigmoid(mask_logits).cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h))

    # BƯỚC B: MASKING & ENHANCEMENT (CLAHE)
    masked = cv2.bitwise_and(raw_img, raw_img, mask=mask)
    lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # BƯỚC C: TRÍCH XUẤT EMBEDDING BẰNG ARCFACE
    arc_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    final_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    final_tensor = arc_transform(final_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = arcface(final_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


# --- 4. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    unet = load_unet()
    arcface = load_arcface()

    # Giả sử bạn lấy 1 tấm làm 'Gốc' và các tấm còn lại để 'Test'
    image_files = [os.path.join(MY_EYES_DIR, f) for f in os.listdir(MY_EYES_DIR) if f.endswith('.jpg')]

    if len(image_files) < 2:
        print("Cần ít nhất 2 ảnh để so sánh!")
    else:
        # Lấy ảnh đầu tiên làm mẫu (Reference)
        ref_emb = get_iris_embedding(image_files[0], unet, arcface)
        print(f"--- Đã đăng ký mẫu từ: {os.path.basename(image_files[0])} ---")

        for i in range(1, len(image_files)):
            test_emb = get_iris_embedding(image_files[i], unet, arcface)
            # Tính Cosine Similarity
            score = torch.mm(ref_emb, test_emb.t()).item()

            result = "KHỚP (MATCH)" if score > THRESHOLD else "KHÔNG KHỚP (UNKNOWN)"
            print(f"Ảnh: {os.path.basename(image_files[i])} | Score: {score:.4f} | Kết quả: {result}")