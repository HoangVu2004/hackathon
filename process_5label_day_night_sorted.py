import json
import os
import shutil
import csv
from pathlib import Path
from PIL import Image, ImageDraw

WORKSPACE = Path("/home/osf/Desktop/hackathon")
DATASET_BASE = WORKSPACE / "datasets" / "mapillary" / "mapillary-vistas-dataset_public_v2.0"
FIVE_LABEL_DIR = WORKSPACE / "5label"

CLASS_NAMES = [
    "Car", "Bus", "Truck", "Other vehicle", "Motorcycle",
    "Motorcyclist", "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"
]

CLASS_COLORS = [
    (0, 0, 255),      # 0: Car (Blue)
    (0, 255, 255),    # 1: Bus (Cyan)
    (255, 165, 0),    # 2: Truck (Orange)
    (128, 0, 128),    # 3: Other vehicle (Purple)
    (255, 0, 255),    # 4: Motorcycle (Magenta)
    (255, 20, 147),   # 5: Motorcyclist (Deep Pink)
    (0, 255, 0),      # 6: Bicycle (Green)
    (50, 205, 50),    # 7: Cyclist (Lime Green)
    (255, 0, 0),      # 8: Pedestrian (Red)
    (255, 255, 0),    # 9: Sign (Yellow)
    (0, 191, 255),    # 10: Traffic light (Deep Sky Blue)
]

def find_original_split(img_stem):
    if (DATASET_BASE / "training" / "images" / f"{img_stem}.jpg").exists():
        return "training"
    elif (DATASET_BASE / "validation" / "images" / f"{img_stem}.jpg").exists():
        return "validation"
    return None

def process_partition(items, partition_name):
    print(f"\n🚀 Đang xử lý tập '{partition_name}' ({len(items)} ảnh, đã sắp xếp theo nhãn cao nhất trước)...")
    target_dir = FIVE_LABEL_DIR / partition_name
    img_dir = target_dir / "images"
    lbl_dir = target_dir / "labels"
    mask_dir = target_dir / "masks"
    mask_col_dir = target_dir / "masks_color"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_col_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    manifest_json = []
    image_paths_list = []

    for rank, item in enumerate(items, 1):
        stem = item['stem']
        src_img = item['src_img']
        src_lbl = item['src_lbl']

        # 1. Copy image
        dst_img = img_dir / f"{stem}.jpg"
        shutil.copy2(src_img, dst_img)
        image_paths_list.append(str(dst_img.resolve()))

        # 2. Copy YOLO-seg label txt
        dst_lbl = lbl_dir / f"{stem}.txt"
        shutil.copy2(src_lbl, dst_lbl)

        # 3. Copy original 16-bit instance PNG mask
        orig_split = find_original_split(stem)
        if orig_split:
            src_mask = DATASET_BASE / orig_split / "v2.0" / "instances" / f"{stem}.png"
            if src_mask.exists():
                shutil.copy2(src_mask, mask_dir / f"{stem}.png")

        # 4. Generate colored overlay mask for 11 classes
        with open(src_lbl, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        with Image.open(src_img) as im:
            w, h = im.size

        color_mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(color_mask)

        for line in lines:
            parts = line.split()
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
            if len(pts) >= 3:
                r, g, b = CLASS_COLORS[cid]
                draw.polygon(pts, fill=(r, g, b, 180), outline=(r, g, b, 255))

        color_mask.save(mask_col_dir / f"{stem}.png")

        # Collect record
        class_names_list = [CLASS_NAMES[c] for c in sorted(item['classes'])]
        row = {
            "rank": rank,
            "image_id": stem,
            "unique_classes_count": item['n_classes'],
            "total_instances_count": item['n_instances'],
            "classes_present": ", ".join(class_names_list),
            "overall_brightness": round(item['overall_mean'], 2),
            "sky_brightness": round(item['sky_mean'], 2),
            "night_score": round(item['night_score'], 2)
        }
        manifest_rows.append(row)
        manifest_json.append(row)

    # Write manifest.csv
    csv_path = target_dir / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "image_id", "unique_classes_count", "total_instances_count",
            "classes_present", "overall_brightness", "sky_brightness", "night_score"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write manifest.json
    with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_json, f, indent=2)

    # Write images.txt (ordered list of paths)
    with open(target_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(image_paths_list) + "\n")

    print(f"✅ Hoàn tất '{partition_name}': 200 ảnh, 200 labels, 200 masks.")
    print(f"   - Hạng #1: {manifest_rows[0]['image_id']} ({manifest_rows[0]['unique_classes_count']} lớp, {manifest_rows[0]['total_instances_count']} instances)")
    print(f"   - Hạng #200: {manifest_rows[-1]['image_id']} ({manifest_rows[-1]['unique_classes_count']} lớp, {manifest_rows[-1]['total_instances_count']} instances)")

def main():
    with open(WORKSPACE / "ranked_scores.json", "r") as f:
        scores = json.load(f)

    # Scan exact label counts in 5label
    img_data = {}
    for split in ["train", "val"]:
        lbl_dir = FIVE_LABEL_DIR / "labels" / split
        img_dir = FIVE_LABEL_DIR / "images" / split
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            stem = fname[:-4]
            lbl_file = lbl_dir / fname
            img_file = img_dir / f"{stem}.jpg"
            with open(lbl_file, "r") as lf:
                lines = [l.strip() for l in lf if l.strip()]
            classes = {int(l.split()[0]) for l in lines}
            img_data[stem] = {
                "stem": stem,
                "src_img": img_file,
                "src_lbl": lbl_file,
                "n_classes": len(classes),
                "n_instances": len(lines),
                "classes": classes
            }

    enriched = []
    for s in scores:
        stem = s['name'][:-4]
        if stem in img_data:
            item = dict(s)
            item.update(img_data[stem])
            enriched.append(item)

    # NIGHT: 200 darkest images (score <= 88.11), sorted by highest label first (n_classes desc, n_instances desc)
    night_pool = enriched[:200]
    night_pool.sort(key=lambda x: (x['n_classes'], x['n_instances']), reverse=True)

    # DAY: Daytime images (score >= 100), sorted by highest label first (n_classes desc, n_instances desc), top 200
    day_pool = [x for x in enriched if x['night_score'] >= 100]
    day_pool.sort(key=lambda x: (x['n_classes'], x['n_instances']), reverse=True)
    top_day_200 = day_pool[:200]

    # Process and save to 5label/night and 5label/day
    process_partition(night_pool, "night")
    process_partition(top_day_200, "day")

    # Create data.yaml inside 5label
    data_yaml_day_night = f"""# YOLO-seg dataset: Day and Night 200 subsets inside 5label
path: {FIVE_LABEL_DIR.resolve()}
train: day/images
val: night/images

nc: 11

names:
  0: Car
  1: Bus
  2: Truck
  3: Other vehicle
  4: Motorcycle
  5: Motorcyclist
  6: Bicycle
  7: Cyclist
  8: Pedestrian
  9: Sign
  10: Traffic light
"""
    with open(FIVE_LABEL_DIR / "day_night_data.yaml", "w") as f:
        f.write(data_yaml_day_night)

    print("\n🎉 HOÀN THÀNH TOÀN BỘ!")

if __name__ == "__main__":
    main()
