import json
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

WORKSPACE = Path("/home/osf/Desktop/hackathon")
DATASET_BASE = WORKSPACE / "datasets" / "mapillary" / "mapillary-vistas-dataset_public_v2.0"
FIVE_LABEL_DIR = WORKSPACE / "5label"
OUTPUT_DIR = WORKSPACE / "5label_day_night"

CLASS_NAMES = [
    "Car", "Bus", "Truck", "Other vehicle", "Motorcycle",
    "Motorcyclist", "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"
]

# Color palette for 11 classes for mask visualization (R, G, B)
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

def process_set(items, subset_name):
    print(f"\nProcessing {subset_name} ({len(items)} images)...")
    sub_dir = OUTPUT_DIR / subset_name
    img_out = sub_dir / "images"
    lbl_out = sub_dir / "labels"
    mask_out = sub_dir / "masks"
    mask_vis_out = sub_dir / "masks_color"

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    mask_vis_out.mkdir(parents=True, exist_ok=True)

    metadata = []

    for rank, item in enumerate(items, 1):
        img_name = item['name']
        stem = os.path.splitext(img_name)[0]
        src_img = item['path']
        src_lbl = item['label_path']

        # 1. Copy image
        dst_img = img_out / img_name
        shutil.copy2(src_img, dst_img)

        # 2. Copy YOLO-seg label txt
        dst_lbl = lbl_out / f"{stem}.txt"
        shutil.copy2(src_lbl, dst_lbl)

        # 3. Copy original 16-bit instance PNG mask
        orig_split = find_original_split(stem)
        if orig_split:
            src_mask = DATASET_BASE / orig_split / "v2.0" / "instances" / f"{stem}.png"
            if src_mask.exists():
                shutil.copy2(src_mask, mask_out / f"{stem}.png")

        # 4. Generate colored mask for 11 target classes
        with open(src_lbl, "r") as f:
            yolo_lines = f.readlines()

        # Open image to get dimensions
        with Image.open(src_img) as im:
            w, h = im.size

        # Create color mask
        color_mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(color_mask)

        for line in yolo_lines:
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
            if len(pts) >= 3:
                r, g, b = CLASS_COLORS[cid]
                # fill with semi-transparent color
                draw.polygon(pts, fill=(r, g, b, 180), outline=(r, g, b, 255))

        # Save color mask PNG
        color_mask.save(mask_vis_out / f"{stem}.png")

        metadata.append({
            "rank": rank,
            "image_id": stem,
            "overall_brightness": round(item['overall_mean'], 2),
            "sky_brightness": round(item['sky_mean'], 2),
            "instances_count": len(yolo_lines)
        })

    with open(sub_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Finished {subset_name}: 200 images, labels, and masks generated.")

def main():
    with open("/home/osf/Desktop/hackathon/ranked_scores.json", "r") as f:
        results = json.load(f)

    # Darkest 200 -> Night
    night_200 = results[:200]
    # Brightest 200 -> Day
    day_200 = results[-200:][::-1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process_set(night_200, "night")
    process_set(day_200, "day")

    # Create data.yaml
    yaml_content = f"""# YOLO-seg dataset: Day and Night subsets (200 images each, >= 5 classes)
path: {OUTPUT_DIR.resolve()}
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
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    # Create classes.txt
    with open(OUTPUT_DIR / "classes.txt", "w") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    print("\n🎉 ALL DONE! Day & Night datasets created in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
