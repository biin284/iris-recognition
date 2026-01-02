# --- save_identity.py (Bản sửa lỗi đồng bộ Crop) ---
import cv2
import torch
import os
import numpy as np
from app_iris import init_system, extract_v4, DEVICE, CROP_RATIO


def fix_and_save():
    unet, arcface = init_system()
    all_embs = []
    ref_dir = 'my_data/Long'

    for f in os.listdir(ref_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(ref_dir, f))

            # --- QUAN TRỌNG: Phải cắt ảnh y hệt như trong App ---
            h, w = img.shape[:2]
            img_cropped = img[0:int(h * CROP_RATIO), 0:w]
            # --------------------------------------------------

            emb, _, _, _ = extract_v4(img_cropped, unet, arcface)
            if emb is not None:
                all_embs.append(emb)

    if all_embs:
        ref_embedding = torch.mean(torch.stack(all_embs), dim=0)
        torch.save(ref_embedding, 'long_identity.pt')
        print("✅ Đã cập nhật long_identity.pt với logic CROP mới!")


if __name__ == "__main__":
    fix_and_save()