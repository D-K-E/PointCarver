# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE
# Tests the main functionality of point carver


import sys
sys.path.insert(0, '..')


from main.pointcarver import SeamMarker
from main.utils import readImage, readPoints, parsePoints, stripExt
from main.utils import qt_image_to_array
from PIL import Image, ImageQt, ImageOps
import unittest
import numpy as np
import os
import json
import pdb

def loadJfile(path):
    "Load json file to memory"
    with open(path, 'r', encoding='utf-8') as f:
        jfile = json.load(f)
    #
    return jfile


def getPointListFromPointPath(ppath) -> [[int, int]]:
    "Get point list from ppath which should be a json file"
    jfile = loadJfile(ppath)
    #
    plist = [[point['y'], point['x']] for point in jfile]
    return plist


def getCoordsDict(coordpath: str):
    "Get coordinate dict with points as keys from coord path"
    jfile = loadJfile(coordpath)
    coords = {tuple(jf['point']): np.array(
        jf['markCoordinates'], dtype=np.int) for jf in jfile}
    return coords


def getCoordArrayFromPath(coordpath):
    "Get coordinate array from path"
    coords = getCoordsDict(coordpath)
    #
    coords = [np.array(coord, dtype=np.int) for coord in coords.values()]
    return coords


def getCoordPair(coordpath, colSlice: bool):
    "Get coordinate pair from coordpath and image path"
    img = np.zeros((2, 2), dtype=np.uint8)
    marker = SeamMarker(img)
    coords_down = loadJfile(coordpath)
    coords = {
            tuple(coord['point']): np.array(coord['markCoordinates']) for coord
            in coords_down
        }
    plist = list(coords.keys())
    pairs = marker.makePairsFromPoints(plist, colSlice=colSlice)
    pair = pairs[0]
    return pair




class PointCarverTest(unittest.TestCase):
    "Test point carver"

    def setUp(self):
        "set up the pointcarver class"
        currentdir = os.getcwd()
        assetdir = os.path.join(currentdir, 'assets')
        self.assetdir = assetdir
        self.imagedir = os.path.join(assetdir, 'images')
        jsondir = os.path.join(assetdir, 'jsonfiles')
        self.npdir = os.path.join(assetdir, 'numpyfiles')
        self.image_path = os.path.join(self.imagedir, 'vietHard.jpg')
        self.emap_path = os.path.join(self.imagedir, 'vietEmap.png')
        self.coords_down_path = os.path.join(jsondir,
                                             'vietHard-coordinates-down.json')
        self.coords_up_path = os.path.join(jsondir,
                                           'vietHard-coordinates-up.json')
        self.points_down_path = os.path.join(jsondir,
                                             'vietHard-points-down.json')
        self.points_up_path = os.path.join(jsondir,
                                           'vietHard-points-up.json')
        self.thresh_val = 5

    def loadImage(self):
        "load and return a copy of the image in the image path"
        pilim = Image.open(self.image_path)
        imarr = np.array(pilim)
        return imarr.copy()

    def test_seammarker_calc_energy(self):
        "tests the calc energy function of pointcarver"
        vietImg = Image.open(self.image_path)
        vietEmap = Image.open(self.emap_path)
        vietEmap = ImageOps.grayscale(vietEmap)
        vietImg = np.array(vietImg, dtype=np.uint8)
        vietEmap = np.array(vietEmap, dtype=np.uint8)
        #
        vietImcp = vietImg.copy()
        vietEmapcp = vietEmap.copy()
        #
        carver = SeamMarker(img=vietImcp)
        emap = carver.calc_energy(vietImcp)
        emap = np.interp(emap,
                         (emap.min(), emap.max()),
                         (0, 256))
        emap = np.uint8(emap)
        comparray = emap == vietEmapcp
        result = comparray.all()
        self.assertTrue(result, "Point carver energy calculation function")

    def test_seammarker_minimum_seam_emap_matrix(self):
        "tests the minimum seam function of pointcarver"
        matrixPath = os.path.join(self.npdir, "vietSliceMatrix.npy")
        compmatrix = np.load(matrixPath)
        vietImcp = self.loadImage()
        vietslice = vietImcp[:, 550:600]
        carver = SeamMarker(img=vietImcp)
        emap = carver.calc_energy(vietslice)
        mat, backtrack = carver.minimum_seam(img=vietslice, emap=emap)
        compmat = mat == compmatrix
        result = compmat.all()
        self.assertTrue(
            result,
            "Point carver minimum seam function emap given, checking matrix"
        )

    def test_seammarker_minimum_seam_emap_backtrack(self):
        backtrackPath = os.path.join(self.npdir, 'vietSliceBacktrack.npy')
        compBacktrack = np.load(backtrackPath)

        vietcp = self.loadImage()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        emap = carver.calc_energy(vietslice)
        mat, backtrack = carver.minimum_seam(img=vietslice, emap=emap)
        compback = backtrack == compBacktrack
        result = compback.all()
        self.assertTrue(
            result,
            "Point carver minimum seam function emap given, checking backtrack"
        )

    def test_seammarker_minimum_seam_backtrack(self):
        backtrackPath = os.path.join(self.npdir, 'vietSliceBacktrack.npy')
        compBacktrack = np.load(backtrackPath)

        vietcp = self.loadImage()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        mat, backtrack = carver.minimum_seam(img=vietslice)
        compback = backtrack == compBacktrack
        result = compback.all()
        self.assertTrue(
            result,
            "Point carver minimum seam function emap not given, "
            "checking backtrack")

    def test_seammarker_mark_column(self):
        compimpath = os.path.join(self.imagedir, 'slicemark.png')
        viet = self.loadImage()
        sliceImage = np.array(Image.open(compimpath), dtype=np.uint8)
        vietcp = viet.copy()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        slicp = vietslice.copy()
        imcp, mask = carver.mark_column(slicp)
        compmark = imcp == sliceImage
        result = compmark.all()
        self.assertTrue(
            result,
            "Point carver mark column function emap not given, "
            "checking if function produces same marks on same slice")

    def test_seammarker_expandPointCoordinate_normal(self):
        viet = self.loadImage()
        vietcp = viet.copy()
        carver = SeamMarker(img=vietcp)
        col_nb = vietcp.shape[1]
        points = getPointListFromPointPath(self.points_down_path)
        point = points[0]
        pointCol = point[1]
        colBefore, colAfter = carver.expandPointCoordinate(
            col_nb,
            coord=pointCol, thresh=self.thresh_val)
        # colnb_comp = 872
        colbef_comp, col_after_comp = 138, 180
        message = "Point column coordinate is expanded to {0} "\
            "in an unexpected way"
        self.assertEqual(colBefore, colbef_comp,
                         message.format('left'))
        self.assertEqual(colAfter, col_after_comp,
                         message.format('right'))

    def test_seammarker_expandPointCoordinate_colBeforeIsZero(self):
        "Coord after is above ubound"
        viet = self.loadImage()
        vietcp = viet.copy()
        carver = SeamMarker(img=vietcp)
        colBefore, colAfter = carver.expandPointCoordinate(80,
                                                           coord=1,
                                                           thresh=5)
        colbef_comp, col_after_comp = 0, 3
        message = "Point column coordinate is expanded to {0} "
        message1 = "minimum column before is normally lower than 0"
        message2 = "maximum column after is not normally"
        self.assertEqual(colBefore, colbef_comp,
                         message.format('left') + message1)
        self.assertEqual(colAfter, col_after_comp,
                         message.format('right') + message2)

    def test_seammarker_matchMarkCoordPairLength_down_colSliceTrue(self):
        colSlice = True
        pair = getCoordPair(self.coords_down_path, colSlice)
        coords = getCoordsDict(self.coords_down_path)
        point1 = pair[0]
        point2 = pair[1]
        marker = SeamMarker(img=np.zeros((2,2), dtype=np.uint8))
        coord1 = coords[point1]
        coord2 = coords[point2]
        coord1_2d = coord1[:, :2]
        coord2_2d = coord2[:, :2]
        uni1 = np.unique(coord1_2d, axis=0)
        uni2 = np.unique(coord2_2d, axis=0)
        retval1, retval2 = marker.matchMarkCoordPairLength(uni1, uni2, colSlice)
        # pdb.set_trace()
        self.assertEqual(retval1.shape, retval2.shape)

    def test_seammarker_swapAndSliceMarkCoordPair_down_colSliceTrue(self):
        ""
        viet = self.loadImage()
        m1 = os.path.join(self.npdir,
                                'matchedMarkCoordPair1ColSliceTrueDown.npy')
        m1 = np.load(m1)
        m2 = os.path.join(self.npdir,
                                'matchedMarkCoordPair2ColSliceTrueDown.npy')
        m2 = np.load(m2)
        marker = SeamMarker(viet)
        colSlice = True
        swaped = marker.swapAndSliceMarkCoordPair(m1, m2,
                                                  viet, colSlice)
        compimg = os.path.join(self.imagedir, 'slicedImage.png')
        compimg = np.array(Image.open(compimg))
        comp = compimg == swaped
        result = comp.all()
        self.assertTrue(result, 
                        "Image slicing with mark coordinate has failed")

    def test_seammarker_sliceImageWithMarkCoordPair_down_colSliceTrue(self):
        "test seam marker slice Image with mark coordinate pair"
        viet = self.loadImage()
        marker = SeamMarker(img=np.zeros((2, 2), dtype=np.uint8))
        coords = getCoordsDict(self.coords_down_path)
        colSlice = True
        pair = getCoordPair(self.coords_down_path, colSlice)
        p1 = pair[0]
        p2 = pair[1]
        coord1 = coords[p1]
        coord2 = coords[p2]
        coord1_2d = coord1[:, :2]
        coord2_2d = coord2[:, :2]
        uni1 = np.unique(coord1_2d, axis=0)
        uni2 = np.unique(coord2_2d, axis=0)
        segment = marker.sliceImageWithMarkCoordPair(viet.copy(),
                                                   uni1,
                                                   uni2,
                                                   colSlice)
        compimg = os.path.join(self.imagedir, 'slicedImage.png')
        compimg = np.array(Image.open(compimg))
        comp = compimg == segment
        result = comp.all()
        self.assertTrue(result, 
                        "Image slicing with mark coordinate has failed")



if __name__ == "__main__":
    unittest.main()
