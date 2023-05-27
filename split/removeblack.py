import os

from PIL import Image

if __name__ == "__main__":
    for f in os.listdir("../dataset"):
        ext = os.path.splitext(f)[1]
        img = Image.open(os.path.join("../dataset", f))
        imgpx = list(img.getdata())
        sum = 0
        min = 0
        max = 0
        for i in imgpx:
            sum = sum + i
        if 1000000 <= sum <= 8000000:
            img.save(os.path.join("../dataset2", f))
