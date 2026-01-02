import cv2
import torch
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
# Import các hàm từ file hiện tại của bạn
from app_iris import init_system, extract_v4, DEVICE

def save_my_vector():
    print("⌛ Đang khởi tạo mô hình để trích xuất...")
    unet, arcface = init_system()

    all_embs = []
    ref_dir = 'my_data/Long'  # Thư mục chứa 7 ảnh mẫu của bạn

    for f in os.listdir(ref_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(ref_dir, f))
            # Trích xuất vector 512-D
            emb, _, status, _ = extract_v4(img, unet, arcface)
            if emb is not None:
                all_embs.append(emb)
                print(f"✅ Đã trích xuất: {f}")

    if all_embs:
        # Tính vector trung bình (Centroid)
        ref_embedding = torch.mean(torch.stack(all_embs), dim=0)
        torch.save(ref_embedding, 'long_identity.pt')
        print("Xong! File 'long_identity.pt' đã sẵn sàng để deploy.")
    else:
        print("Lỗi: Không tìm thấy ảnh mẫu hợp lệ.")

if __name__ == "__main__":
    save_my_vector()