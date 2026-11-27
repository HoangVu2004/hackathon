import json
import os
import shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

WORKSPACE = Path("/home/osf/Desktop/hackathon")
DATASET_BASE = WORKSPACE / "datasets" / "mapillary" / "mapillary-vistas-dataset_public_v2.0"
OUTPUT_DIR = WORKSPACE / "5label"

MAPPING = {
    'object--vehicle--car': 0,
    'object--vehicle--bus': 1,
    'object--vehicle--truck': 2,
    'object--vehicle--other-vehicle': 3,
    'object--vehicle--caravan': 3,
    'object--vehicle--trailer': 3,
    'object--vehicle--wheeled-slow': 3,
    'object--vehicle--on-rails': 3,
    'object--vehicle--boat': 3,
    'object--vehicle--motorcycle': 4,
    'human--rider--motorcyclist': 5,
    'object--vehicle--bicycle': 6,
    'human--rider--bicyclist': 7,
    'human--person--individual': 8,
    'object--traffic-sign--front': 9,
    'object--traffic-sign--direction-front': 9,
    'object--traffic-sign--temporary-front': 9,
    'object--traffic-sign--information-parking': 9,
    'object--traffic-sign--ambiguous': 9,
    'object--sign--information': 9,
    'object--sign--other': 9,
    'object--sign--advertisement': 9,
    'object--sign--store': 9,
    'object--traffic-light--general-single': 10,
    'object--traffic-light--pedestrians': 10,
    'object--traffic-light--general-upright': 10,
    'object--traffic-light--general-horizontal': 10,
    'object--traffic-light--cyclists': 10,
    'object--traffic-light--other': 10,
}

CLASS_NAMES = [
    "Car", "Bus", "Truck", "Other vehicle", "Motorcycle",
    "Motorcyclist", "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"
]

def process_single_item(item):
    split, json_path, out_split = item
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", [])
        width = data.get("width")
        height = data.get("height")

        classes_present = set()
        yolo_lines = []
        class_counts = Counter()

        for obj in objects:
            lbl = obj.get("label")
            if lbl in MAPPING:
                cid = MAPPING[lbl]
                classes_present.add(cid)
                class_counts[cid] += 1
                poly = obj.get("polygon", [])
                norm_coords = []
                for pt in poly:
                    nx = max(0.0, min(1.0, pt[0] / width))
                    ny = max(0.0, min(1.0, pt[1] / height))
                    norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
                if norm_coords:
                    yolo_lines.append(f"{cid} " + " ".join(norm_coords) + "\n")

        # Check condition: AT LEAST 5 UNIQUE CLASSES
        if len(classes_present) >= 5:
            img_id = os.path.basename(json_path)[:-5]
            src_img = DATASET_BASE / split / "images" / f"{img_id}.jpg"
            
            dst_img_dir = OUTPUT_DIR / "images" / out_split
            dst_lbl_dir = OUTPUT_DIR / "labels" / out_split

            # Write label txt
            dst_lbl = dst_lbl_dir / f"{img_id}.txt"
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                lf.writelines(yolo_lines)

            # Copy image file
            dst_img = dst_img_dir / f"{img_id}.jpg"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)

            return True, len(classes_present), len(yolo_lines), dict(class_counts)
        else:
            return False, len(classes_present), 0, {}
    except Exception as e:
        return False, 0, 0, {}

def main():
    splits = [
        ("training", "train"),
        ("validation", "val")
    ]

    for _, out_s in splits:
        (OUTPUT_DIR / "images" / out_s).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / out_s).mkdir(parents=True, exist_ok=True)

    items = []
    for split, out_s in splits:
        poly_dir = DATASET_BASE / split / "v2.0" / "polygons"
        for fname in os.listdir(poly_dir):
            if fname.endswith(".json"):
                items.append((split, os.path.join(poly_dir, fname), out_s))

    print(f"Bắt đầu xử lý {len(items)} files bằng đa tiến trình...")

    total_kept = 0
    total_instances = 0
    class_stats = Counter()
    class_count_dist = Counter()

    with ProcessPoolExecutor() as executor:
        for kept, n_classes, n_inst, counts in executor.map(process_single_item, items, chunksize=100):
            if kept:
                total_kept += 1
                total_instances += n_inst
                class_count_dist[n_classes] += 1
                for cid, cval in counts.items():
                    class_stats[cid] += cval

    print(f"\n✅ Đã lọc thành công:")
    print(f"- Tổng số ảnh giữ lại (>= 5 nhãn): {total_kept}/{len(items)}")
    print(f"- Tổng số đối tượng mask (YOLO-seg): {total_instances}")
    print("\nPhân bố số lượng lớp trên mỗi ảnh giữ lại:")
    for n in sorted(class_count_dist.keys(), reverse=True):
        print(f"  {n} lớp: {class_count_dist[n]} ảnh")

    print("\nThống kê số lượng đối tượng theo từng class:")
    for cid, name in enumerate(CLASS_NAMES):
        print(f"  [{cid:2d}] {name:16s}: {class_stats[cid]} instances")

    # Tạo data.yaml cho thư mục 5label
    data_yaml_content = f"""# YOLO-seg configuration for 5label filtered dataset
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val

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
    with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    # Tạo classes.txt
    with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    print("\n✅ Đã tạo data.yaml và classes.txt trong thư mục '5label'!")

if __name__ == "__main__":
    main()
