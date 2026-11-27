import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

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

CLASS_NAMES = {
    0: 'Car',
    1: 'Bus',
    2: 'Truck',
    3: 'Other vehicle',
    4: 'Motorcycle',
    5: 'Motorcyclist',
    6: 'Bicycle',
    7: 'Cyclist',
    8: 'Pedestrian',
    9: 'Sign',
    10: 'Traffic light'
}

def analyze_file(item):
    split, filepath = item
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        classes_present = set()
        class_counts = Counter()
        for obj in data.get('objects', []):
            lbl = obj.get('label')
            if lbl in MAPPING:
                cid = MAPPING[lbl]
                classes_present.add(cid)
                class_counts[cid] += 1
        img_id = os.path.basename(filepath)[:-5]
        return split, img_id, filepath, len(classes_present), sum(class_counts.values()), dict(class_counts)
    except Exception as e:
        return split, '', filepath, 0, 0, {}

def main():
    base = '/home/osf/Desktop/hackathon/datasets/mapillary/mapillary-vistas-dataset_public_v2.0'
    all_files = []
    for split in ['training', 'validation']:
        d = f'{base}/{split}/v2.0/polygons'
        if os.path.exists(d):
            for fname in os.listdir(d):
                if fname.endswith('.json'):
                    all_files.append((split, os.path.join(d, fname)))

    print(f'Total json files to scan: {len(all_files)}')

    with ProcessPoolExecutor() as ex:
        results = list(ex.map(analyze_file, all_files, chunksize=100))

    by_n_classes = Counter([r[3] for r in results])
    print('\nDistribution of unique target class counts per image:')
    for k in sorted(by_n_classes.keys(), reverse=True):
        print(f'  {k} unique classes: {by_n_classes[k]} images')

    # Sort by number of classes descending, then number of instances descending
    sorted_res = sorted(results, key=lambda x: (x[3], x[4]), reverse=True)

    max_classes = sorted_res[0][3]
    print(f'\nMaximum unique classes found in a single image: {max_classes}')

    top_matches = [r for r in sorted_res if r[3] == max_classes]
    print(f'Total images having {max_classes} unique classes: {len(top_matches)}')

    for i, (split, img_id, fpath, n_cls, n_inst, counts) in enumerate(top_matches[:10], 1):
        c_names = [CLASS_NAMES[c] for c in sorted(counts.keys())]
        print(f'{i}. [{split}] {img_id}: {n_cls} classes, {n_inst} instances -> {c_names}')

    return sorted_res

if __name__ == '__main__':
    main()
