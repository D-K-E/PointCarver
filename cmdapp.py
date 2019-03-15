# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE

from main.pointcarver import SeamMarker
from main.utils import readImage, readPoints, parsePoints

from PIL import Image


def getArgs_proc():
    "Get arguments from user"
    print("Hello and welcome to pointcarver interface")
    print("Please make sure you have the following in your hands:\n")
    print("    ", "- Path to your image")
    print("    ", "- Path to a list of reference points for the carver")
    print("    ", "  File containing points should have the following structure:")
    print("    ", "- y1, x1\n      y2,x2\n      y3, x3\n      ...\n")
    print("    ", "- Direction for the carver")
    print("    ", "- Threshold for segmentation area. It should be thought as")
    print("    ", "  rough measure of how far lines/columns are from each other.")
    print("    ", "- Path to save segmented parts\n")
    impath = input("Please enter full path to image: ")
    pointpath = input("Please enter full path to point file: ")
    direction = input(
        "Please enter a direction in lower case. Available directions are:\n"\
        "down, up, left, right [down by default]: "
                      )
    if direction not in ["down", 'up', "left", "right"]:
        direction = "down"
    threshold = input("Please enter a threshold [10 by default]: ")
    if theshold = "":
        threshold = 10
    else:
        threshold = int(threshold)
    savepath = input("Please enter full path to save segmented parts: ")
    image = readImage(impath)
    points = readPoints(pointpath)
    return image, points, direction, threshold, savepath


if __name__ == "__main__":
    img, ps, direct, thresh, spath = getArgs_proc()
    marker = SeamMarker(img=img,
                        plist=ps,
                        thresh=thresh,
                        direction=direct)
    segments = marker.segmentWithPoints()
    for i, seg in enumerate(segments):
        pilim = Image.fromarray(seg)
        name = str(i) + ".png"
        savename = os.path.join(spath, name)
        pilim.save(savename)
