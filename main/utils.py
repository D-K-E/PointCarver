from PIL import Image
import numpy as np


def stripExt(str1: str, ext_delimiter='.') -> [str, str]:
    "Strip extension"
    strsplit = str1.split(ext_delimiter)
    ext = strsplit.pop()
    newstr = ext_delimiter.join(strsplit)
    return (newstr, ext)


def readImage(path: str) -> np.ndarray:
    "Read image from the path"
    pilim = Image.open(path)
    return np.array(pilim)


def parsePoints(pointstr: str) -> list:
    "Parse string which contains points into a point list"
    # the points are y1, x1; y2, x2; y3, x3;
    points = pointstr.splitlines()
    points = [
        [int(p.strip()) for p in po.split(',') if p] for po in points if po
    ]
    return points


def readPoints(path):
    "Read points from given path"
    with open('r', encoding="utf-8") as f:
        pointstr = f.read()
    points = parsePoints(pointstr)
    return points

