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
IDS_DIR = 'identities'  # Thư mục chứa dữ liệu định danh
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
            torch.nn.Linear(960, 512),
            torch.nn.BatchNorm1d(512)
        )

    def forward(self, x): return self.backbone(x)


@st.cache_resource
def init_system():
    # Load U-Net
    unet = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=1, activation=None).to(DEVICE)
    if os.path.exists(UNET_PATH):
        unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))

    # Load ArcFace
    arcface = IrisArcFace().to(DEVICE)
    if os.path.exists(ARCFACE_PATH):
        sd = torch.load(ARCFACE_PATH, map_location=DEVICE)
        arcface.load_state_dict({k: v for k, v in sd.items() if 'head' not in k}, strict=False)

    unet.eval()
    arcface.eval()
    return unet, arcface


# --- 3. CÁC HÀM XỬ LÝ DỮ LIỆU ---

def load_identity_db():
    """Tải toàn bộ dữ liệu người dùng từ thư mục identities"""
    db = {}
    if not os.path.exists(IDS_DIR):
        os.makedirs(IDS_DIR)
        return db

    for f in os.listdir(IDS_DIR):
        if f.endswith('.pt'):
            # Tên file dạng "long_identity.pt" -> Lấy tên "Long"
            name = f.replace("_identity.pt", "").capitalize()
            path = os.path.join(IDS_DIR, f)
            try:
                embedding = torch.load(path, map_location=DEVICE)
                db[name] = embedding
            except Exception as e:
                st.error(f"Lỗi tải file {f}: {e}")
    return db


def save_embedding(name, embedding):
    """Lưu hoặc cập nhật vector đặc trưng của người dùng"""
    if not os.path.exists(IDS_DIR):
        os.makedirs(IDS_DIR)

    # Chuẩn hóa tên file (chữ thường, không dấu cách)
    safe_name = name.lower().strip().replace(" ", "_")
    filename = f"{safe_name}_identity.pt"
    save_path = os.path.join(IDS_DIR, filename)

    # Nếu đã có file cũ -> Cộng trung bình để cập nhật (Học tăng cường)
    if os.path.exists(save_path):
        old_emb = torch.load(save_path, map_location=DEVICE)
        new_emb = (old_emb + embedding) / 2.0
        new_emb = F.normalize(new_emb, p=2, dim=1)  # Chuẩn hóa lại sau khi cộng
    else:
        new_emb = embedding

    torch.save(new_emb, save_path)
    return True


def extract_v4(frame, unet, arcface):
    """Trích xuất đặc trưng mống mắt từ ảnh"""
    h, w = frame.shape[:2]
    roi_area = h * w
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (256, 256))
    img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_input).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = (torch.sigmoid(unet(img_tensor)).cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h))

    # Tìm vùng mống mắt
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt, max_area = None, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        # Lọc nhiễu
        if circularity > MIN_CIRCULARITY and area > max_area:
            max_area, best_cnt = area, cnt

    if best_cnt is None: return None, 0, "KHÔNG TÌM THẤY MẮT", None

    # Cắt và Tăng cường ảnh
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    l, a, b = cv2.split(cv2.cvtColor(masked, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Trích xuất Vector
    pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    tensor = arc_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = F.normalize(arcface(tensor), p=2, dim=1)

    return embedding, max_area / roi_area, "OK", enhanced


# --- 4. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="Iris Guard v4", page_icon="🛡️")
st.title("🛡️ Hệ thống Nhận diện Mống mắt")

# Menu bên trái
menu = ["Chấm công (Xác thực)", "Đăng ký mới"]
choice = st.sidebar.selectbox("Chức năng", menu)
st.sidebar.info("Hạn chót đồ án: 03/01/2026")

# Khởi tạo model 1 lần
unet, arcface = init_system()

# --- MÀN HÌNH ĐĂNG KÝ ---
if choice == "Đăng ký mới":
    st.header("📝 Đăng ký nhân viên mới")
    st.write("Hãy nhập tên và chụp ảnh để tạo dữ liệu mẫu.")

    new_name = st.text_input("Nhập tên nhân viên (Không dấu, VD: Anh, Long):")

    # Chụp ảnh đăng ký
    reg_img = st.camera_input("Chụp ảnh mẫu (Nên chụp 2-3 lần ở các góc sáng khác nhau)")

    if reg_img and new_name:
        bytes_data = np.asarray(bytearray(reg_img.read()), dtype=np.uint8)
        frame = cv2.imdecode(bytes_data, 1)

        with st.spinner("Đang trích xuất đặc trưng..."):
            emb, ratio, status, enhanced_img = extract_v4(frame, unet, arcface)

        if emb is None:
            st.error(f"Không thể lấy mẫu: {status}. Vui lòng chụp lại!")
            st.image(frame, caption="Ảnh lỗi", width=300)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(enhanced_img, caption="Mống mắt trích xuất", channels="BGR")
            with col2:
                st.success("Trích xuất thành công!")
                if st.button("Lưu dữ liệu này"):
                    save_embedding(new_name, emb)
                    st.toast(f"Đã lưu dữ liệu cho: {new_name}!", icon="✅")
                    st.info("Mẹo: Bạn có thể chụp tiếp tấm nữa và bấm Lưu để làm dữ liệu phong phú hơn.")

# --- MÀN HÌNH CHẤM CÔNG ---
elif choice == "Chấm công (Xác thực)":
    st.header("clock in/out")

    # Load lại DB mỗi lần vào màn hình này để cập nhật người mới
    identity_db = load_identity_db()

    if not identity_db:
        st.warning("Chưa có dữ liệu nhân viên. Vui lòng sang tab 'Đăng ký mới' để tạo dữ liệu.")
    else:
        st.success(f"Hệ thống đã sẵn sàng với {len(identity_db)} nhân viên: {', '.join(identity_db.keys())}")

        check_img = st.camera_input("Quét mống mắt để chấm công")

        if check_img:
            bytes_data = np.asarray(bytearray(check_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(bytes_data, 1)

            emb, ratio, status, enhanced_img = extract_v4(frame, unet, arcface)

            if emb is None:
                st.warning(f"Lỗi: {status}")
            else:
                # --- LOGIC SO KHỚP 1-N ---
                best_score = -1.0
                best_name = "Unknown"

                for name, ref_emb in identity_db.items():
                    score = torch.mm(emb, ref_emb.t()).item()
                    if score > best_score:
                        best_score = score
                        best_name = name

                # Hiển thị
                col1, col2 = st.columns(2)
                with col1:
                    st.image(frame, caption="Ảnh quét", channels="BGR")
                with col2:
                    st.image(enhanced_img, caption="Mống mắt", channels="BGR")

                st.divider()

                if best_score > THRESHOLD:
                    st.success(f"✅ XÁC THỰC THÀNH CÔNG: {best_name.upper()}")
                    st.metric(label="Độ tin cậy", value=f"{best_score:.4f}", delta="Hợp lệ")
                    st.balloons()
                else:
                    st.error("❌ TỪ CHỐI TRUY CẬP")
                    st.write(f"Người giống nhất: {best_name} ({best_score:.4f}) - Dưới ngưỡng {THRESHOLD}")