#!/usr/bin/env python3
"""
Tự động lọc nhãn và re-map class cho dataset YOLO-seg (Instance Segmentation).
Giữ lại 11 lớp theo yêu cầu:
  0: Car
  1: Bus
  2: Truck
  3: Other vehicle
  4: Motorcycle (không người)
  5: Motorcyclist (có người)
  6: Bicycle (không người)
  7: Cyclist (có người)
  8: Pedestrian
  9: Sign
  10: Traffic light
"""

import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# CẤU HÌNH THƯ MỤC
# ==============================================================================
WORKSPACE = Path("/home/osf/Desktop/hackathon")
SRC_DIR = WORKSPACE / "datasets" / "raw"        # Thư mục chứa dataset gốc
DST_DIR = WORKSPACE / "datasets" / "filtered"   # Thư mục chứa dataset sau khi lọc

# ==============================================================================
# BẢNG ÁNH XẠ (MAPPING TABLE)
# Cú pháp: { old_class_id_hoac_name: new_class_id }
# Bạn chỉ cần điền ID cũ trong prelabel tương ứng với các ID mới từ 0 đến 10:
# ==============================================================================
CLASS_NAMES = [
    "Car",            # 0
    "Bus",            # 1
    "Truck",          # 2
    "Other vehicle",  # 3
    "Motorcycle",     # 4
    "Motorcyclist",   # 5
    "Bicycle",        # 6
    "Cyclist",        # 7
    "Pedestrian",     # 8
    "Sign",           # 9
    "Traffic light"   # 10
]

# ĐIỀN ID HOẶC TÊN TƯƠNG ỨNG CỦA DATASET GỐC VÀO ĐÂY:
OLD_TO_NEW_MAP = {
    # Ví dụ: nếu prelabel cũ có id 0 là car -> map sang 0
    # 0: 0,
    # Nếu prelabel cũ có id 5 là bus -> map sang 1
    # 5: 1,
}

def process_split(split_name="train", keep_empty=False):
    src_img_dir = SRC_DIR / "images" / split_name
    src_lbl_dir = SRC_DIR / "labels" / split_name

    # Nếu không chia split train/val thì quét trực tiếp
    if not src_lbl_dir.exists():
        src_img_dir = SRC_DIR / "images"
        src_lbl_dir = SRC_DIR / "labels"
        dst_img_dir = DST_DIR / "images"
        dst_lbl_dir = DST_DIR / "labels"
    else:
        dst_img_dir = DST_DIR / "images" / split_name
        dst_lbl_dir = DST_DIR / "labels" / split_name

    if not src_lbl_dir.exists():
        print(f"⚠️  Không tìm thấy thư mục nhãn: {src_lbl_dir}")
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(list(src_lbl_dir.glob("*.txt")))
    print(f"\n📂 Đang xử lý tập '{split_name}': {len(label_files)} files nhãn...")

    stats = {i: 0 for i in range(len(CLASS_NAMES))}
    kept_images = 0
    total_objects = 0

    for lbl_path in tqdm(label_files, desc=f"Processing {split_name}"):
        filtered_lines = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    old_id = int(parts[0])
                except ValueError:
                    continue

                if old_id in OLD_TO_NEW_MAP:
                    new_id = OLD_TO_NEW_MAP[old_id]
                    coords = parts[1:]
                    filtered_lines.append(f"{new_id} " + " ".join(coords) + "\n")
                    stats[new_id] += 1
                    total_objects += 1

        # Nếu có nhãn hợp lệ (hoặc muốn giữ ảnh rỗng)
        if len(filtered_lines) > 0 or keep_empty:
            # Ghi file nhãn
            out_lbl = dst_lbl_dir / lbl_path.name
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.writelines(filtered_lines)

            # Copy ảnh tương ứng
            img_copied = False
            for ext in [".jpg", ".png", ".jpeg", ".webp", ".JPG", ".PNG"]:
                img_path = src_img_dir / f"{lbl_path.stem}{ext}"
                if img_path.exists():
                    shutil.copy2(img_path, dst_img_dir / img_path.name)
                    img_copied = True
                    break

            if img_copied:
                kept_images += 1

    print(f"\n✅ Hoàn thành tập '{split_name}':")
    print(f"  - Số ảnh giữ lại: {kept_images}/{len(label_files)}")
    print(f"  - Tổng số đối tượng mask: {total_objects}")
    for cid, name in enumerate(CLASS_NAMES):
        print(f"    [{cid:2d}] {name:16s}: {stats[cid]}")

if __name__ == "__main__":
    if not OLD_TO_NEW_MAP:
        print("⚠️  CHÚ Ý: Bạn chưa điền OLD_TO_NEW_MAP trong file filter_labels.py.")
        print("   Hãy mở file và thêm các ID cũ tương ứng vào dictionary OLD_TO_NEW_MAP.")
        print("   Ví dụ: OLD_TO_NEW_MAP = {2: 0, 5: 1, ...}")
    else:
        for split in ["train", "val", "test"]:
            process_split(split)
