import cv2
import torch
import os
import numpy as np
from app_iris import init_system, extract_v4, DEVICE, CROP_MARGIN


def fix_and_save():
    print("⌛ Khởi tạo hệ thống...")
    unet, arcface = init_system()
    all_embs = []
    ref_dir = 'my_data/Long'

    if not os.path.exists(ref_dir):
        print(f"❌ Thư mục {ref_dir} không tồn tại!")
        return

    for f in os.listdir(ref_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(ref_dir, f))
            if img is None: continue

            # Center-Crop đồng bộ 100% với App
            h, w = img.shape[:2]
            h_start, h_end = int(h * CROP_MARGIN), int(h * (1 - CROP_MARGIN))
            img_cropped = img[h_start:h_end, 0:w]

            emb, _, _, _ = extract_v4(img_cropped, unet, arcface)
            if emb is not None:
                all_embs.append(emb)
                print(f"✅ Đã trích xuất thành công: {f}")

    if all_embs:
        ref_embedding = torch.mean(torch.stack(all_embs), dim=0)
        torch.save(ref_embedding, 'long_identity.pt')
        print("\n✨ Xong! Đã tạo long_identity.pt")
    else:
        print("❌ Lỗi: Không trích xuất được đặc trưng từ ảnh mẫu.")


if __name__ == "__main__":
    fix_and_save()