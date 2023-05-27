import os
import warnings

import PIL
from PIL import Image
from itertools import product


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    tile("mapwrld.png", "./", "./", 512)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
