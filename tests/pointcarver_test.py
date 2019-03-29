# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE
# Tests the main functionality of point carver

from ..main.pointcarver import SeamMarker
from ..main.utils import readImage, readPoints, parsePoints, stripExt
from ..main.utils import qt_image_to_array

from PIL import Image, ImageQt
import unittest
import numpy as np
import os
import json


def getPointListFromPointPath(ppath) -> [[int,int]]:
    "Get point list from ppath which should be a json file"
    with open(ppath, 'r', encoding='utf-8') as f:
        jfile = json.load(f)
    #
    plist = [[point['y'], point['x']] for point in jfile]
    return plist


def getCoordArrayFromPath(coordpath):
    "Get coordinate array from path"
    with open(coordpath, 'r', encoding='utf-8') as f:
        jfile = json.load(f)
        coords = {jf['point']: np.array(jf['markCoordinates'], dtype=np.int) for jf in jfile}
    #
    coords = [np.array(coord, dtype=np.int) for coord in coords]



class PointCarverTest(unittest.TestCase):
    "Test point carver"
    def setUp(self):
        "set up the pointcarver class"
        currentdir = os.getcwd()
        self.image_path = os.path.join(assetdir, 'vietHard.jpg')
        assetdir = os.path.join(currentdir, 'assets')
        self.coords_down_path = os.path.join(assetdir, 
                                             'vietHard-coordinates-down.json')
        self.coords_up_path = os.path.join(assetdir, 
                                             'vietHard-coordinates-up.json')
        self.points_down_path = os.path.join(assetdir,
                                             'vietHard-points-down.json')
        self.points_down_path = os.path.join(assetdir,
                                             'vietHard-points-up.json')
        self.thresh_val = 5
    
