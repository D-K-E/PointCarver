from PIL import Image
import numpy as np
import json

from PySide2 import QtGui

# From stack overflow

def saveJson(path, obj):
    "Save json"
    with open(path, 'w', 
              encoding='utf-8', newline='\n') as f:
        json.dump(obj, f)

def qt_image_to_array(img: QtGui.QImage):
    """ Creates a numpy array from a QImage.

        If share_memory is True, the numpy array and the QImage is shared.
        Be careful: make sure the numpy array is destroyed before the image, 
        otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QtGui.QImage), "img must be a QtGui.QImage object"
    assert img.format() == QtGui.QImage.Format.Format_RGB32,\
        "img format must be QImage.Format.Format_RGB32, got: {}".format(
            img.format())

    height = img.height()
    width = img.width()
    channels = img.depth()
    bbuffer = img.constBits()

    # Sanity check
    n_bits_buffer = len(bbuffer) * 8
    n_bits_image = width * height * channels
    assert n_bits_buffer == n_bits_image,\
        "size mismatch: {} != {}".format(n_bits_buffer, n_bits_image)

    assert img.depth() == 32, "unexpected image depth: {}".format(img.depth())

    # Note the different width height parameter order!
    arr = np.ndarray(shape=(height,
                            width,
                            channels // 8
                            ),
                     buffer=bbuffer,
                     dtype=np.uint8)
    return arr # change rgb -> bgr

# end stack overflow



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

