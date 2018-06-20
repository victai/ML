from sys import argv
from PIL import Image

if len(argv) >= 2:
    im = Image.open(argv[1])
#pix = im.load()
width, height = im.size

for y in range(height):
    for x in range(width):
        pix = im.getpixel((x, y))
        pix = (int(pix[0]/2),
               int(pix[1]/2),
               int(pix[2]/2))
        im.putpixel((x, y), pix)
im.save('Q2.png')
#im.show()
