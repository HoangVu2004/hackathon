import glob
import os
from PIL import Image, ImageStat
from concurrent.futures import ProcessPoolExecutor

def get_brightness(p):
    try:
        im = Image.open(p).convert('L').resize((64, 64))
        stat = ImageStat.Stat(im)
        return p, stat.mean[0], stat.rms[0]
    except Exception as e:
        return p, 0, 0

def main():
    imgs = glob.glob('/home/osf/Desktop/hackathon/5label/images/train/*.jpg') + \
           glob.glob('/home/osf/Desktop/hackathon/5label/images/val/*.jpg')
    print(f'Total images in 5label: {len(imgs)}')

    with ProcessPoolExecutor() as ex:
        results = list(ex.map(get_brightness, imgs, chunksize=100))

    results.sort(key=lambda x: x[1])

    print(f'Darkest 10 images (likely night):')
    for p, m, rms in results[:10]:
        print(f'  {os.path.basename(p)} -> mean: {m:.2f}, rms: {rms:.2f}')

    print(f'\nBrightest 10 images (day):')
    for p, m, rms in results[-10:]:
        print(f'  {os.path.basename(p)} -> mean: {m:.2f}, rms: {rms:.2f}')

    # Check distribution
    under_40 = sum(1 for r in results if r[1] < 40)
    under_50 = sum(1 for r in results if r[1] < 50)
    under_60 = sum(1 for r in results if r[1] < 60)
    over_100 = sum(1 for r in results if r[1] > 100)
    over_120 = sum(1 for r in results if r[1] > 120)

    print(f'\nStats:')
    print(f'  mean < 40: {under_40}')
    print(f'  mean < 50: {under_50}')
    print(f'  mean < 60: {under_60}')
    print(f'  mean > 100: {over_100}')
    print(f'  mean > 120: {over_120}')

if __name__ == '__main__':
    main()
