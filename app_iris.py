import streamlit as st
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os

# --- 1. CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_PATH = 'iris_unet_v3.pth'
ARCFACE_PATH = 'iris_arcface_best.pth'
ID_PATH = 'long_identity.pt'
THRESHOLD = 0.44  #
MIN_IRIS_RATIO = 0.035
MIN_CIRCULARITY = 0.55
CROP_MARGIN = 0.1

arc_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 2. LOAD MÔ HÌNH & VECTOR ---
class IrisArcFace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(960, 512), torch.nn.BatchNorm1d(512)
        )

    def forward(self, x): return self.backbone(x)


@st.cache_resource
def init_system():
    unet = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=1, activation=None).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    arcface = IrisArcFace().to(DEVICE)
    sd = torch.load(ARCFACE_PATH, map_location=DEVICE)
    arcface.load_state_dict({k: v for k, v in sd.items() if 'head' not in k}, strict=False)
    unet.eval();
    arcface.eval()
    return unet, arcface


@st.cache_data
def get_ref_embedding():
    if os.path.exists(ID_PATH):
        return torch.load(ID_PATH, map_location=DEVICE)
    return None


# --- 3. XỬ LÝ V4 ---
def extract_v4(frame, unet, arcface):
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
    if not contours: return None, 0, "KHÔNG TÌM THẤY MẮT", None

    best_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < roi_area * 0.005: continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        if circularity > MIN_CIRCULARITY:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    if best_cnt is None: return None, 0, "LỖI HÌNH DÁNG (SHAPE ERR)", None
    ratio = max_area / roi_area
    if ratio < MIN_IRIS_RATIO: return None, ratio, f"MẮT QUÁ NHỎ ({ratio:.1%})", None

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    tensor = arc_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = F.normalize(arcface(tensor), p=2, dim=1)
    return embedding, ratio, "OK", enhanced


# --- 4. GIAO DIỆN ---
st.set_page_config(page_title="Iris Guard v4 Mobile")
st.title("🛡️ Hệ thống Nhận diện Mống mắt")

unet, arcface = init_system()
ref_emb = get_ref_embedding()

img_file = st.camera_input("Chụp ảnh mống mắt")

if img_file and ref_emb is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame_original = cv2.imdecode(file_bytes, 1)

    # Cắt ảnh
    h, w = frame_original.shape[:2]
    h_start, h_end = int(h * CROP_MARGIN), int(h * (1 - CROP_MARGIN))
    frame_cropped = frame_original[h_start:h_end, 0:w]

    emb, ratio, status, enhanced_img = extract_v4(frame_cropped, unet, arcface)

    if emb is None:
        st.warning(f"Trạng thái: {status}")
        st.image(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB), caption="Vùng ảnh đang quét")
    else:
        score = torch.mm(emb, ref_emb.t()).item()
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB), caption="Vùng xử lý")
        with col2:
            st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), caption="Mống mắt")

        if score > THRESHOLD:
            st.success(f"✅ HỢP LỆ (Score: {score:.4f})")
            st.balloons()
        else:
            st.error(f"❌ KHÔNG KHỚP (Score: {score:.4f})")