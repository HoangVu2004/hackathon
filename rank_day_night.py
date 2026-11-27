import glob
import os
import json
from PIL import Image, ImageStat
from concurrent.futures import ProcessPoolExecutor

def analyze_img(item):
    p, lbl_path = item
    try:
        im = Image.open(p)
        # Use draft for fast loading
        im.draft('L', (128, 128))
        im_gray = im.convert('L').resize((128, 128))
        
        # Overall mean brightness
        stat = ImageStat.Stat(im_gray)
        overall_mean = stat.mean[0]
        
        # Top 25% (sky region)
        sky_crop = im_gray.crop((0, 0, 128, 32))
        sky_stat = ImageStat.Stat(sky_crop)
        sky_mean = sky_stat.mean[0]

        # Night score: lower overall_mean and lower sky_mean = more definitely night
        # Night metric: 0.5 * overall_mean + 0.5 * sky_mean
        night_score = 0.5 * overall_mean + 0.5 * sky_mean

        return {
            'path': p,
            'label_path': lbl_path,
            'name': os.path.basename(p),
            'overall_mean': overall_mean,
            'sky_mean': sky_mean,
            'night_score': night_score
        }
    except Exception as e:
        return None

def main():
    train_imgs = glob.glob('/home/osf/Desktop/hackathon/5label/images/train/*.jpg')
    val_imgs = glob.glob('/home/osf/Desktop/hackathon/5label/images/val/*.jpg')
    all_imgs = train_imgs + val_imgs
    
    items = []
    for p in all_imgs:
        split = 'train' if 'train' in p else 'val'
        stem = os.path.splitext(os.path.basename(p))[0]
        lbl = f'/home/osf/Desktop/hackathon/5label/labels/{split}/{stem}.txt'
        items.append((p, lbl))

    print(f'Analyzing {len(items)} images...')
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(analyze_img, items, chunksize=100))

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x['night_score'])

    print(f'\n--- TOP 10 NIGHT CANDIDATES ---')
    for r in results[:10]:
        print(f"{r['name']}: overall={r['overall_mean']:.1f}, sky={r['sky_mean']:.1f}, score={r['night_score']:.1f}")

    print(f'\n--- 200th NIGHT CANDIDATE ---')
    r200 = results[199]
    print(f"{r200['name']}: overall={r200['overall_mean']:.1f}, sky={r200['sky_mean']:.1f}, score={r200['night_score']:.1f}")

    print(f'\n--- TOP 10 DAY CANDIDATES (Brightest) ---')
    for r in results[-10:]:
        print(f"{r['name']}: overall={r['overall_mean']:.1f}, sky={r['sky_mean']:.1f}, score={r['night_score']:.1f}")

    print(f'\n--- 200th DAY CANDIDATE (from brightest) ---')
    r_day200 = results[-200]
    print(f"{r_day200['name']}: overall={r_day200['overall_mean']:.1f}, sky={r_day200['sky_mean']:.1f}, score={r_day200['night_score']:.1f}")

    with open('/home/osf/Desktop/hackathon/ranked_scores.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
