# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE
# Tests the utils functionality of seam marker

from seammarker.utils import shapeCoordinate
from seammarker.utils import normalizeImageVals
from seammarker.utils import getConsecutive1D
from seammarker.utils import getConsecutive2D
from seammarker.utils import getLinesFromCoordinates
from seammarker.utils import saveJson
from seammarker.utils import stripExt
from seammarker.utils import readImage
from seammarker.utils import assertCond
from seammarker.utils import drawMark2Image

import unittest
import os
from PIL import Image, ImageOps
import numpy as np
import json
import pdb


def drawCoordinate(img: np.ndarray,
                   coordarr: np.ndarray):
    "Draw coordinate"
    imcp = img.copy()
    for i in range(coordarr.shape[0]):
        r, c = coordarr[i, :]
        imcp[r, c] = 255
    return imcp


class UtilsTest(unittest.TestCase):
    "Test utility functions of seam marking"

    def setUp(self):
        "set up the pointcarver class"
        currentdir = str(__file__)
        currentdir = os.path.join(currentdir, os.pardir)
        currentdir = os.path.join(currentdir, os.pardir)
        currentdir = os.path.abspath(currentdir)
        testdir = os.path.join(currentdir, "tests")
        assetdir = os.path.join(testdir, 'assets')
        self.assetdir = assetdir
        self.imagedir = os.path.join(assetdir, 'images')
        self.utilsImageDir = os.path.join(self.imagedir, 'utils')
        jsondir = os.path.join(assetdir, 'jsonfiles')
        self.npdir = os.path.join(assetdir, 'numpyfiles')
        self.image_col_path = os.path.join(self.imagedir, 'vietHard.jpg')
        self.image_row_path = os.path.join(self.imagedir, 'demotik.png')
        self.emap_path = os.path.join(self.imagedir, 'vietEmap.png')
        self.coords_down_path = os.path.join(jsondir,
                                             'vietHard-coordinates-down.json')
        self.coords_up_path = os.path.join(jsondir,
                                           'vietHard-coordinates-up.json')
        self.coords_left_path = os.path.join(jsondir,
                                             "demotik-coordinates-left.json")
        self.points_down_path = os.path.join(jsondir,
                                             'vietHard-points-down.json')
        self.points_up_path = os.path.join(jsondir,
                                           'vietHard-points-up.json')
        self.points_left_path = os.path.join(jsondir,
                                             'demotik-points-left.json')

    def compareArrays(self, arr1, arr2, message):
        "Compare arrays for equality"
        result = arr1 == arr2
        result = result.all()
        self.assertTrue(result, message)

    def test_getConsecutive1d(self):
        "test function for obtaining consecutive values in an array"
        inarr = np.array([0, 1, 123, 15, 16, 17, 138, 124,
                          186, 139, 125, 126])
        g1 = np.array([0, 1])
        g2 = np.array([15, 16, 17])
        g3 = np.array([125, 126])
        outarr = [g1, g2, g3]
        only_index = False
        result, indices = getConsecutive1D(inarr, only_index=only_index)
        subarrays = []
        for res in result:
            if res.size != 1:
                subarrays.append(res)
        for i in range(len(subarrays)):
            sub = subarrays[i]
            out = outarr[i]
            self.compareArrays(sub, out,
                               "Array " + str(i) + " is not the same with"
                               " compare value.")

    def test_getConsecutive2dHorizontal(self):
        "Obtain consecutive values from 2d array"
        inarr = np.array([
            [42, 2], [42, 3],  # y1, x1 and y2, x2
            [42, 4], [150, 123],
            [160, 123], [160, 124],
            [160, 186], [161, 125],
            [186, 126], [188, 127],
            [197, 135], [199, 186]
        ])

        g1 = np.array([[42, 2], [42, 3], [42, 4]])
        g2 = np.array([[160, 123], [160, 124]])
        outarr = [g1, g2]
        consvals, indices = getConsecutive2D(inarr,
                                             direction="horizontal",
                                             only_index=False)
        for i in range(len(consvals)):
            val = consvals[i]
            out = outarr[i]
            self.compareArrays(val, out,
                               "Array " + str(i) + " is not the same with"
                               " compare value.")

    def test_getConsecutive2dVertical(self):
        "Obtain consecutive values from 2d array"
        inarr = np.array([
            [2, 42], [3, 42],  # y1, x1 and y2, x2
            [4, 42], [150, 123],
            [123, 160], [124, 160],
            [160, 186], [161, 125],
            [186, 126], [188, 127],
            [197, 135], [199, 186]
        ])
        g1 = np.array([[2, 42], [3, 42], [4, 42]])
        g2 = np.array([[123, 160], [124, 160]])
        outarr = [g1, g2]
        consvals, indices = getConsecutive2D(inarr,
                                             direction="vertical",
                                             only_index=False)
        for i in range(len(consvals)):
            val = consvals[i]
            out = outarr[i]
            self.compareArrays(val, out,
                               "Array " + str(i) + " is not the same with"
                               " compare value.")

    def test_getConsecutive2dDiagonalL(self):
        "Obtain consecutive values from 2d array"
        inarr = np.array([
            [2, 42], [3, 43],  # y1, x1 and y2, x2
            [4, 44], [150, 123],
            [123, 161], [124, 162],
            [160, 186], [161, 125],
            [186, 126], [188, 127],
            [197, 135], [199, 186]
        ])
        g1 = np.array([[2, 42], [3, 43], [4, 44]])
        g2 = np.array([[123, 161], [124, 162]])
        outarr = [g1, g2]
        consvals, indices = getConsecutive2D(inarr,
                                             direction="diagonal-l",
                                             only_index=False)
        for i in range(len(consvals)):
            val = consvals[i]
            out = outarr[i]
            self.compareArrays(val, out,
                               "Array " + str(i) + " is not the same with"
                               " compare value.")

    def test_getConsecutive2dDiagonalR(self):
        "Obtain consecutive values from 2d array"
        inarr = np.array([
            [2, 44], [3, 43],  # y1, x1 and y2, x2
            [4, 42], [150, 123],
            [123, 162], [124, 161],
            [160, 186], [161, 125],
            [186, 126], [188, 127],
            [197, 135], [199, 186]
        ])
        g1 = np.array([[2, 44], [3, 43], [4, 42]])
        g2 = np.array([[123, 162], [124, 161]])
        outarr = [g1, g2]
        consvals, indices = getConsecutive2D(inarr,
                                             direction="diagonal-r",
                                             only_index=False)
        for i in range(len(consvals)):
            val = consvals[i]
            out = outarr[i]
            self.compareArrays(val, out,
                               "Array " + str(i) + " is not the same with"
                               " compare value.")

    def test_getLinesFromCoordinates(self):
        vietImg = np.array(Image.open(self.image_col_path))
        vietslice = vietImg[:, 550:600]
        vietimg = Image.fromarray(vietslice)
        grayimg = ImageOps.grayscale(vietimg)
        vietimg = np.array(grayimg)
        vietimg = normalizeImageVals(vietimg)
        data = np.zeros((vietimg.shape[0], vietimg.shape[1], 3), dtype=np.int)
        data[:, :, 0] = vietimg
        rnb, cnb = vietslice.shape[:2]
        arr = np.array([[[r, c] for c in range(cnb)] for r in range(rnb)],
                       dtype=np.int)
        data[:, :, 1:] = arr
        datacond = data[:, :, 0] < data[:, :, 0].mean()
        data = data[datacond, :]
        data = data[:, 1:]
        # arr = arr.reshape((-1, 2))
        # arr = np.unique(arr, axis=0)
        lines = getLinesFromCoordinates(data)
        for i, line in enumerate(lines):
            lineimg = drawCoordinate(vietslice, line)
            imname = "imline-" + str(i) + ".png"
            impath = os.path.join(self.utilsImageDir, imname)
            imarr = np.array(Image.open(impath))
            self.compareArrays(imarr, lineimg,
                               "comparison of line image " + str(i) + " failed"
                               )


if __name__ == "__main__":
    unittest.main()
