from PIL import Image
import numpy as np
import json

# From stack overflow


def saveJson(path, obj):
    "Save json"
    with open(path, 'w',
              encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def shapeCoordinate(coord: np.ndarray):
    "Reshape coordinate to have [[y, x]] structure"
    cshape = coord.shape
    assert 2 in cshape or 3 in cshape
    if cshape[0] == 2:
        coord = coord.T
    elif cshape[1] == 2:
        pass
    elif cshape[0] == 3:
        coord = coord.T
        coord = coord[:, :2]
    elif cshape[1] == 3:
        coord = coord[:, :2]
    # obtain unique coords
    uni1, index = np.unique(coord, return_index=True, axis=0)
    uni1 = coord[np.sort(index), :]
    return uni1

# Debug related


def drawMark2Image(image: np.ndarray,
                   coord: np.ndarray,
                   imstr: str):
    zeroimg = np.zeros_like(image, dtype=np.uint8)
    imcp = image.copy()
    assert coord.shape[1] == 2
    for i in range(coord.shape[0]):
        yx = coord[i, :]
        imcp[yx[0], yx[1], :] = 255
        zeroimg[yx[0], yx[1], :] = 255
    #
    zeroname = imstr + "-zero.png"
    name = imstr + ".png"
    Image.fromarray(imcp).save(name)
    Image.fromarray(zeroimg).save(zeroname)
    return imcp, zeroimg
