import cv2
import torch
import os
import numpy as np
from app_iris import init_system, extract_v4


def save_clean_identity():
    print("⌛ Khởi tạo...")
    unet, arcface = init_system()
    all_embs, ref_dir = [], 'my_data/Long'

    for f in os.listdir(ref_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(ref_dir, f))
            # Không Crop, trích xuất đặc trưng trực tiếp từ ảnh gốc
            emb, _, _, _ = extract_v4(img, unet, arcface)
            if emb is not None:
                all_embs.append(emb)
                print(f"✅ OK: {f}")

    if all_embs:
        torch.save(torch.mean(torch.stack(all_embs), dim=0), 'long_identity.pt')
        print("\n✨ Đã tạo long_identity.pt (Không Crop).")


if __name__ == "__main__":
    save_clean_identity()