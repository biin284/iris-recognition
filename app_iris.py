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
IDS_DIR = 'identities'  # Thư mục chứa các file .pt của nhiều người
THRESHOLD = 0.222  # Ngưỡng an toàn
MIN_IRIS_RATIO = 0.020
MIN_CIRCULARITY = 0.40

arc_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 2. LOAD MÔ HÌNH ---
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
    if os.path.exists(UNET_PATH):
        unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))

    arcface = IrisArcFace().to(DEVICE)
    if os.path.exists(ARCFACE_PATH):
        sd = torch.load(ARCFACE_PATH, map_location=DEVICE)
        arcface.load_state_dict({k: v for k, v in sd.items() if 'head' not in k}, strict=False)

    unet.eval()
    arcface.eval()
    return unet, arcface


# --- HÀM MỚI: LOAD TẤT CẢ DANH TÍNH TỪ THƯ MỤC ---
@st.cache_data
def load_identity_db():
    db = {}
    if not os.path.exists(IDS_DIR):
        os.makedirs(IDS_DIR)
        return db

    # Quét tất cả file .pt trong thư mục identities
    for f in os.listdir(IDS_DIR):
        if f.endswith('.pt'):
            # Lấy tên từ tên file (vd: long_identity.pt -> Long)
            name = f.replace("_identity.pt", "").capitalize()
            path = os.path.join(IDS_DIR, f)
            try:
                embedding = torch.load(path, map_location=DEVICE)
                db[name] = embedding
            except Exception as e:
                st.error(f"Lỗi tải file {f}: {e}")
    return db


def extract_v4(frame, unet, arcface):
    h, w = frame.shape[:2]
    roi_area = h * w
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (256, 256))
    img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_input).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = (torch.sigmoid(unet(img_tensor)).cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt, max_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity > MIN_CIRCULARITY and area > max_area:
            max_area, best_cnt = area, cnt

    if best_cnt is None: return None, 0, "KHÔNG TÌM THẤY MẮT", None

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    l, a, b = cv2.split(cv2.cvtColor(masked, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    tensor = arc_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = F.normalize(arcface(tensor), p=2, dim=1)
    return embedding, max_area / roi_area, "OK", enhanced


# --- 4. GIAO DIỆN ---
st.set_page_config(page_title="Iris Guard v4 Mobile")
st.title("🛡️ Hệ thống Nhận diện Mống mắt (Multi-User)")

# Khởi tạo
unet, arcface = init_system()
identity_db = load_identity_db()

# Hiển thị danh sách người dùng đã nạp
if identity_db:
    st.success(f"Đã tải {len(identity_db)} danh tính: {', '.join(identity_db.keys())}")
else:
    st.error(f"Chưa có dữ liệu danh tính nào trong thư mục '{IDS_DIR}'")

img_file = st.camera_input("Chụp ảnh mống mắt")

if img_file and identity_db:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    emb, ratio, status, enhanced_img = extract_v4(frame, unet, arcface)

    if emb is None:
        st.warning(f"Trạng thái: {status}")
        st.image(frame, channels="BGR")
    else:
        # --- LOGIC SO KHỚP 1-N ---
        best_score = -1.0
        best_name = "Unknown"

        # So sánh embedding hiện tại với TẤT CẢ người trong DB
        for name, ref_emb in identity_db.items():
            score = torch.mm(emb, ref_emb.t()).item()
            if score > best_score:
                best_score = score
                best_name = name
        # -------------------------

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Ảnh gốc")
        with col2:
            st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), caption="Mống mắt trích xuất")

        # Hiển thị kết quả
        if best_score > THRESHOLD:
            st.success(f"✅ XIN CHÀO: {best_name.upper()}")
            st.info(f"Độ tương đồng: {best_score:.4f}")
            st.balloons()
        else:
            st.error(f"❌ NGƯỜI LẠ (Unknown)")
            st.write(f"Kết quả tốt nhất: {best_name} ({best_score:.4f}) - Dưới ngưỡng an toàn")