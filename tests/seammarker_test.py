# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE
# Tests the main functionality of point carver


from pointcarver.src.seammarker import SeamMarker
from pointcarver.src.utils import readImage, readPoints, parsePoints, stripExt
from pointcarver.src.utils import qt_image_to_array
from pointcarver.src.utils import shapeCoordinate
from PIL import Image, ImageQt, ImageOps
import unittest
import numpy as np
import os
import io
import base64
import json
import pdb


def img2base64(img: Image):
    buff = io.BytesIO()
    img.save(buff, format='PNG')
    imagebytes = buff.getvalue()
    if hasattr(base64, 'encodebytes'):
        imstr = base64.encodebytes(imagebytes)
    else:
        imstr = base64.encodestring(imagebytes)
    return imstr


def img2str(img: Image) -> str:
    f = io.BytesIO()
    imbin = img.save(f, format='PNG')
    imbin = f.getvalue()
    return str(imbin, 'latin1', 'strict')


def str2img(imdata: str):
    imbyte = bytes(imdata, 'latin1', 'strict')
    f = io.BytesIO()
    f.write(imbyte)
    return Image.open(f, mode='r')


def loadJfile(path):
    "Load json file to memory"
    with open(path, 'r', encoding='utf-8') as f:
        jfile = json.load(f)
    #
    return jfile


def getPointListFromPointPath(ppath) -> [[int, int]]:
    "Get point list from ppath which should be a json file"
    jfile = loadJfile(ppath)
    # print(jfile)
    # pdb.set_trace()
    #
    plist = [[point['y'], point['x']] for i, point in jfile.items()]
    return plist


def prepPointCoord(ppath: str):
    jfile = loadJfile(ppath)
    fcopy = jfile.copy()
    for i, pdata in jfile.items():
        fcopy[i]['seamCoordinates'] = np.array(pdata['seamCoordinates'],
                                               dtype=np.int)
    #
    return fcopy


def getCoordsDict(coordpath: str):
    "Get coordinate dict with points as keys from coord path"
    jfile = loadJfile(coordpath)
    coords = {
        (coord['y'],
         coord['x']): np.array(coord['seamCoordinates']) for i, coord
        in jfile.items()
    }
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
    coords = getCoordsDict(coordpath)
    plist = list(coords.keys())
    pairs = marker.makePairsFromPoints(plist, colSlice=colSlice)
    return pairs[0]


def getCoordWithPoint(coordpath: str, point: (int, int), xFirst=False):
    "Get coordinate dict using point from coordpath"
    jfile = loadJfile(coordpath)
    for i, pointData in jfile.items():
        if xFirst:
            if point == (pointData['x'], pointData['y']):
                return pointData
        else:
            if point == (pointData['y'], pointData['x']):
                return pointData


class PointCarverTest(unittest.TestCase):
    "Test point carver"

    def setUp(self):
        "set up the pointcarver class"
        currentdir = os.getcwd()
        testdir = os.path.join(currentdir, "tests")
        assetdir = os.path.join(testdir, 'assets')
        self.assetdir = assetdir
        self.imagedir = os.path.join(assetdir, 'images')
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

        self.GEN_SLICE_IMAGE = False
        self.sliceImagePath = os.path.join(self.imagedir,
                                           'slicedImage.png')
        self.GEN_SLICE_MARK = False
        self.slicemarkPath = os.path.join(self.imagedir,
                                          'slicemark.png')
        self.GEN_SLICEROW_MARK = False
        self.sliceRowMarkPath = os.path.join(self.imagedir,
                                             'slicerowmark.png')
        self.GEN_ROW_SLICE = False
        self.rowslicePath = os.path.join(self.imagedir,
                                         'rowslice.png')
        self.GEN_COL_SLICE = False
        self.colslicePath = os.path.join(self.imagedir,
                                         'colslice.png')
        self.GEN_MARK_COLS = False
        self.markColumnsImagePath = os.path.join(self.imagedir,
                                                 'markedColumnsImage.png')
        self.GEN_MARK_ROWS = False
        self.markRowsImagePath = os.path.join(self.imagedir,
                                              'markedRowsImage.png')
        self.GEN_ROW_ADDED = False
        self.rowAddedImagePath = os.path.join(self.imagedir,
                                              "rowAddedImage.png")
        self.GEN_COL_ADDED = False
        self.columnAddedImagePath = os.path.join(self.imagedir,
                                                 "columnAddedImage.png")
        self.GEN_COL_MARKED = False
        self.colMarkedImagePath = os.path.join(self.imagedir,
                                               "colMarkedImage.png")
        self.GEN_ROW_MARKED = False
        self.rowMarkedImagePath = os.path.join(self.imagedir,
                                               "rowMarkedImage.png")
        self.GEN_SEGMENTS = False
        self.segmentPaths = [
            os.path.join(self.imagedir, str(i) + '-viet.png'
                         ) for i in range(28)
        ]
        self.GEN_SLICE_MAT = False
        self.slicemat_path = os.path.join(self.npdir,
                                          'vietSliceMatrix.npy')
        self.GEN_SLICE_BACKTRACK_MAT = False
        self.slicemat_backtrack_path = os.path.join(self.npdir,
                                                    'vietSliceBacktrack.npy')
        self.GEN_MATCH_COORD_PAIRS = False
        self.match_coord_path = os.path.join(
            self.npdir,
            "matchedMarkCoordPair{0}ColSliceTrueDown.npy"
        )

    def generateSliceMatrix(self):
        if self.GEN_SLICE_MAT:
            img = self.loadImageCol()
            viet = np.array(img.copy(), dtype=np.uint8)
            vietslice = viet[:, 550:600]
            carver = SeamMarker(img=None)
            emap = carver.calc_energy(viet.copy())
            mat, backtrack = carver.minimum_seam(img=vietslice.copy(),
                                                 emap=emap)
            np.save(self.slicemat_path, mat)

    def generateBacktrackSliceMatrix(self):
        if self.GEN_SLICE_BACKTRACK_MAT:
            vietcp = self.loadImageCol()
            vietslice = vietcp[:, 550:600]
            carver = SeamMarker(img=vietcp)
            emap = carver.calc_energy(vietslice.copy())
            mat, backtrack = carver.minimum_seam(img=vietslice, emap=emap)
            np.save(self.slicemat_backtrack_path, backtrack)

    def generateMatchCoordinateMatrix(self, colSlice: bool,
                                      isUpTo: bool):
        if self.GEN_MATCH_COORD_PAIRS:
            firstPath = self.match_coord_path.format(str(1))
            secondPath = self.match_coord_path.format(str(2))
            coords = getCoordsDict(self.coords_down_path)
            pair = getCoordPair(self.coords_down_path, colSlice)
            p1 = pair[0]
            p2 = pair[1]
            coord1 = coords[p1]
            coord2 = coords[p2]
            marker = SeamMarker(img=np.zeros((2, 2), dtype=np.uint8))
            markCoord1, markCoord2 = marker.matchMarkCoordPairLength(
                coord1, coord2, colSlice, isUpTo
            )
            np.save(firstPath, markCoord1)
            np.save(secondPath, markCoord2)

    def generateSlicedImage(self):
        if self.GEN_SLICE_IMAGE:
            viet = self.loadImageCol()
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
            Image.fromarray(swaped).save(self.sliceImagePath)

    def generateSliceMark(self):
        if self.GEN_SLICE_MARK:
            viet = self.loadImageCol()
            vietcp = viet.copy()
            vietslice = vietcp[:, 550:600]
            carver = SeamMarker(img=vietcp)
            slicp = vietslice.copy()
            imcp, mask = carver.mark_column(slicp)
            Image.fromarray(imcp).save(self.slicemarkPath)

    def generateSliceRowMark(self):
        if self.GEN_SLICEROW_MARK:
            demot = self.loadImageRow()
            carver = SeamMarker(img=demot)
            clip = demot[150:250, :]
            clip, mask = carver.mark_row(clip)
            Image.fromarray(clip).save(self.sliceRowMarkPath)

    def generateRowSlice(self):
        if self.GEN_ROW_SLICE:
            demot = self.loadImageRow()
            points = getPointListFromPointPath(self.points_left_path)
            point = points[0]
            carver = SeamMarker(demot)
            rowslice = carver.getRowSliceOnPoint(point, demot.copy(),
                                                 isUpTo=True,
                                                 thresh=5)[0]
            Image.fromarray(rowslice).save(self.rowslicePath)

    def generateColSlice(self):
        if self.GEN_COL_SLICE:
            viet = self.loadImageCol()
            points = getPointListFromPointPath(self.points_down_path)
            point = points[0]
            marker = SeamMarker(viet)
            colslice = marker.getColumnSliceOnPoint(point,
                                                    viet,
                                                    thresh=5,
                                                    isUpTo=False)[0]
            Image.fromarray(colslice).save(self.colslicePath)

    def generateMarkedColumnsImage(self):
        if self.GEN_MARK_COLS:
            viet = self.loadImageCol()
            marker = SeamMarker(viet)
            points = loadJfile(self.points_down_path)
            markedImage = marker.markPointListSeam(img=viet,
                                                   plist=points)
            Image.fromarray(markedImage).save(self.markColumnsImagePath)

    def generateMarkedRowsImage(self):
        if self.GEN_MARK_ROWS:
            demot = self.loadImageRow()
            marker = SeamMarker(demot)
            points = loadJfile(self.points_left_path)
            markedImage = marker.markPointListSeam(img=demot,
                                                   plist=points)
            Image.fromarray(markedImage).save(self.markRowsImagePath)

    def generateRowAdded(self):
        if self.GEN_ROW_ADDED:
            isUpTo = True
            demot = self.loadImageRow()
            points = getPointListFromPointPath(self.points_left_path)
            point = points[0]
            marker = SeamMarker(demot)
            before = 0
            after = 62
            rowslice = os.path.join(self.imagedir, 'rowslice.png')
            rowslice = np.array(Image.open(rowslice))
            rowslicecp = rowslice.copy()
            rowslicehalf = rowslicecp.shape[0] // 2
            rowslicecp[rowslicehalf:rowslicehalf+3, :] = 255
            addedimg = marker.addRowSlice2Image(demot,
                                                point,
                                                beforeAfterCoord=(
                                                    before, after),
                                                imgSlice=rowslicecp,
                                                isUpTo=isUpTo)
            Image.fromarray(addedimg).save(self.rowAddedImagePath)

    def generateColumnAdded(self):
        if self.GEN_COL_ADDED:
            isUpTo = False
            viet = self.loadImageCol()
            points = getPointListFromPointPath(self.points_down_path)
            point = points[0]
            marker = SeamMarker(viet)
            before = 138
            after = 180
            compcolslice = os.path.join(self.imagedir, 'colslice.png')
            compimg = np.array(Image.open(compcolslice))
            compcp = compimg.copy()
            compcphalf = compcp.shape[1] // 2
            compcp[:, compcphalf:compcphalf+3] = 255
            # pdb.set_trace()
            addedimg = marker.addColumnSlice2Image(viet,
                                                   point,
                                                   beforeAfterCoord=(before,
                                                                     after),
                                                   imgSlice=compcp,
                                                   isUpTo=isUpTo)
            Image.fromarray(addedimg).save(self.columnAddedImagePath)

    def generateColMarkedImage(self):
        if self.GEN_COL_MARKED:
            isUpTo = False
            colSlice = True
            viet = self.loadImageCol()
            points = loadJfile(self.points_down_path)
            point = points["0"]
            point_coord = (point['y'], point['x'])
            marker = SeamMarker(viet)
            markedImage = marker.markSeam4Point(viet,
                                                point_coord,
                                                isUpTo,
                                                point['threshold'],
                                                colSlice,
                                                mark_color=[0, 255, 0]
                                                )
            Image.fromarray(markedImage).save(self.colMarkedImagePath)

    def generateRowMarkedImage(self):
        if self.GEN_ROW_MARKED:
            isUpTo = True
            colSlice = False
            demot = self.loadImageRow()
            points = loadJfile(self.points_left_path)
            point = points["2"]
            point_coord = (point['y'], point['x'])
            marker = SeamMarker(demot)
            markedImage = marker.markSeam4Point(demot,
                                                point_coord,
                                                isUpTo,
                                                point['threshold'],
                                                colSlice,
                                                mark_color=[0, 255, 0]
                                                )
            Image.fromarray(markedImage).save(self.rowMarkedImagePath)

    def generateSegments(self):
        if self.GEN_SEGMENTS:
            img = self.loadImageCol()
            marker = SeamMarker(img=np.zeros((2, 2), dtype=np.uint8))
            coords = prepPointCoord(self.coords_down_path)
            segments = marker.segmentImageWithPointListSeamCoordinate(
                image=img, coords=coords)
            paths = self.segmentPaths
            [
                [
                    Image.fromarray(segs[i]).save(paths[i]) for i in range(
                        len(segs)
                    )
                ] for direction, segs in segments.items()
            ]

    def loadImage(self, path):
        pilim = Image.open(path)
        imarr = np.array(pilim)
        return imarr.copy()

    def loadImageFromCoordinatePath(self, cpath):
        ""
        jfile = loadJfile(cpath)
        image = jfile['image']
        img = str2img(image['data'])

        return img

    def loadImageCol(self):
        "load and return a copy of the image in the image path"
        return self.loadImage(self.image_col_path)

    def loadImageRow(self):
        ""
        return self.loadImage(self.image_row_path)

    def loadSegments(self):
        ""
        segments = []
        for i in range(28):
            segpath = self.segmentPaths[i]
            seg = np.array(Image.open(segpath))
            segments.append(seg)
        return segments

    def compareArrays(self, arr1, arr2, message):
        "Compare arrays for equality"
        result = arr1 == arr2
        result = result.all()
        self.assertTrue(result, message)

    def test_seammarker_calc_energy(self):
        "tests the calc energy function of pointcarver"
        vietEmap = Image.open(self.emap_path)
        vietEmap = ImageOps.grayscale(vietEmap)
        vietImg = self.loadImageCol()
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
        self.compareArrays(emap, vietEmapcp,
                           "Point carver energy calculation function")

    def test_seammarker_minimum_seam_emap_matrix(self):
        "tests the minimum seam function of pointcarver"
        matrixPath = os.path.join(self.npdir, "vietSliceMatrix.npy")
        compmatrix = np.load(matrixPath)
        vietImcp = self.loadImageCol()
        vietslice = vietImcp[:, 550:600]
        carver = SeamMarker(img=vietImcp)
        emap = carver.calc_energy(vietslice)
        mat, backtrack = carver.minimum_seam(img=vietslice, emap=emap)
        self.compareArrays(mat, compmatrix,
                           "Point carver minimum seam function emap given, checking matrix")

    def test_seammarker_minimum_seam_emap_backtrack(self):
        backtrackPath = os.path.join(self.npdir, 'vietSliceBacktrack.npy')
        compBacktrack = np.load(backtrackPath)

        vietcp = self.loadImageCol()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        emap = carver.calc_energy(vietslice)
        mat, backtrack = carver.minimum_seam(img=vietslice, emap=emap)
        self.compareArrays(
            backtrack, compBacktrack,
            "Point carver minimum seam function emap given, checking backtrack"
        )

    def test_seammarker_minimum_seam_backtrack(self):
        backtrackPath = os.path.join(self.npdir, 'vietSliceBacktrack.npy')
        compBacktrack = np.load(backtrackPath)

        vietcp = self.loadImageCol()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        mat, backtrack = carver.minimum_seam(img=vietslice)
        self.compareArrays(
            backtrack, compBacktrack,
            "Point carver minimum seam function emap not given, "
            "checking backtrack"
        )

    def test_seammarker_mark_column(self):
        compimpath = os.path.join(self.imagedir, 'slicemark.png')
        viet = self.loadImageCol()
        sliceImage = np.array(Image.open(compimpath), dtype=np.uint8)
        vietcp = viet.copy()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        slicp = vietslice.copy()
        imcp, mask = carver.mark_column(slicp)
        self.compareArrays(
            imcp, sliceImage,
            "Point carver mark column function emap not given, "
            "checking if function produces same marks on same slice")

    def test_seammarkerk_mark_row(self):
        demot = self.loadImageRow()
        compimg = os.path.join(self.imagedir, 'slicerowmark.png')
        compimg = np.array(Image.open(compimg))
        carver = SeamMarker(img=demot)
        clip = demot[150:250, :]
        clip, mask = carver.mark_row(clip)
        # pdb.set_trace()
        self.compareArrays(
            clip, compimg,
            "Point carver mark row function emap not given, "
            "checking if function produces same marks on same slice"
        )

    def test_seammarker_expandPointCoordinate_normal(self):
        viet = self.loadImageCol()
        vietcp = viet.copy()
        carver = SeamMarker(img=vietcp)
        col_nb = vietcp.shape[1]
        points = getPointListFromPointPath(self.points_down_path)
        point = points[0]
        pointCol = point[1]
        colBefore, colAfter = carver.expandPointCoordinate(
            col_nb,
            coord=pointCol, thresh=5)
        # colnb_comp = 872
        colbef_comp, col_after_comp = 65, 107
        # pdb.set_trace()
        message = "Point column coordinate is expanded to {0} "\
            "in an unexpected way"
        self.assertEqual(colBefore, colbef_comp,
                         message.format('left'))
        self.assertEqual(colAfter, col_after_comp,
                         message.format('right'))

    def test_seammarker_expandPointCoordinate_colBeforeIsZero(self):
        "Coord after is above ubound"
        viet = self.loadImageCol()
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

    def test_seammarker_getRowSliceOnPointIsUpToTrue(self):
        demot = self.loadImageRow()
        points = getPointListFromPointPath(self.points_left_path)
        point = points[0]
        carver = SeamMarker(demot)
        rowslice = carver.getRowSliceOnPoint(point, demot, isUpTo=True,
                                             thresh=5)[0]
        compimg = os.path.join(self.imagedir, 'rowslice.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(rowslice, compimg,
                           "rowslice is not the same for sliced image"
                           )

    def test_seammarker_getColSliceOnPointIsUpToFalse(self):
        viet = self.loadImageCol()
        points = getPointListFromPointPath(self.points_down_path)
        point = points[0]
        marker = SeamMarker(viet)
        colslice = marker.getColumnSliceOnPoint(point,
                                                viet,
                                                thresh=5,
                                                isUpTo=False)[0]
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'colslice.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(colslice, compimg,
                           "column slice is not the same for sliced image"
                           )

    def test_seammarker_sliceOnPointIsUpToFalseColSliceTrue(self):
        viet = self.loadImageCol()
        points = getPointListFromPointPath(self.points_down_path)
        point = points[0]
        marker = SeamMarker(viet)
        colslice = marker.sliceOnPoint(viet,
                                       point,
                                       thresh=5,
                                       isUpTo=False,
                                       colSlice=True)
        # pdb.set_trace()
        colslice = colslice[0]
        compimg = os.path.join(self.imagedir, 'colslice.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(colslice, compimg,
                           "column slice is not the same for sliced image"
                           )

    def test_seammarker_sliceOnPointIsUpToTrueColSliceFalse(self):
        demot = self.loadImageRow()
        points = getPointListFromPointPath(self.points_left_path)
        point = points[0]
        marker = SeamMarker(demot)
        rowslice = marker.sliceOnPoint(demot,
                                       point,
                                       thresh=5,
                                       isUpTo=True,
                                       colSlice=False)
        rowslice = rowslice[0]
        compimg = os.path.join(self.imagedir, 'rowslice.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(rowslice, compimg,
                           "row slice is not the same for sliced image"
                           )

    def test_seammarker_addColumnSlice2ImageIsUpToFalse(self):
        isUpTo = False
        viet = self.loadImageCol()
        points = getPointListFromPointPath(self.points_down_path)
        point = points[0]
        marker = SeamMarker(viet)
        before = 138
        after = 180
        compcolslice = os.path.join(self.imagedir, 'colslice.png')
        compimg = np.array(Image.open(compcolslice))
        compcp = compimg.copy()
        compcphalf = compcp.shape[1] // 2
        compcp[:, compcphalf:compcphalf+3] = 255
        # pdb.set_trace()
        addedimg = marker.addColumnSlice2Image(viet,
                                               point,
                                               beforeAfterCoord=(before,
                                                                 after),
                                               imgSlice=compcp,
                                               isUpTo=isUpTo)
        compimg = os.path.join(self.imagedir, 'columnAddedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(
            addedimg, compimg,
            "Added column does not give the expected added image"
        )

    def test_seammarker_addRowSlice2ImageIsUpToTrue(self):
        isUpTo = True
        demot = self.loadImageRow()
        points = getPointListFromPointPath(self.points_left_path)
        point = points[0]
        marker = SeamMarker(demot)
        before = 0
        after = 62
        rowslice = os.path.join(self.imagedir, 'rowslice.png')
        rowslice = np.array(Image.open(rowslice))
        rowslicecp = rowslice.copy()
        rowslicehalf = rowslicecp.shape[0] // 2
        rowslicecp[rowslicehalf:rowslicehalf+3, :] = 255
        addedimg = marker.addRowSlice2Image(demot,
                                            point,
                                            beforeAfterCoord=(before, after),
                                            imgSlice=rowslicecp,
                                            isUpTo=isUpTo)
        compimg = os.path.join(self.imagedir, 'rowAddedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(addedimg, compimg,
                           "Added row does not give added image"
                           )

    def test_seammarker_addPointSlice2Image_colSliceTrue_isUpToFalse(self):
        isUpTo = False
        colSlice = True
        viet = self.loadImageCol()
        points = getPointListFromPointPath(self.points_down_path)
        point = points[0]
        marker = SeamMarker(viet)
        before = 138
        after = 180
        compcolslice = os.path.join(self.imagedir, 'colslice.png')
        compimg = np.array(Image.open(compcolslice))
        compcp = compimg.copy()
        compcphalf = compcp.shape[1] // 2
        compcp[:, compcphalf:compcphalf+3] = 255
        addedimg = marker.addPointSlice2Image(viet,
                                              point,
                                              beforeAfterCoord=(before,
                                                                after),
                                              colSlice=colSlice,
                                              imgSlice=compcp,
                                              isUpTo=isUpTo)
        compimg = os.path.join(self.imagedir, 'columnAddedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(
            addedimg, compimg,
            "Added column does not give the expected added image"
        )

    def test_seammarker_addPointSlice2Image_colSliceFalse_isUpToTrue(self):
        isUpTo = True
        colSlice = False
        demot = self.loadImageRow()
        points = getPointListFromPointPath(self.points_left_path)
        point = points[0]
        marker = SeamMarker(demot)
        before = 0
        after = 62
        rowslice = os.path.join(self.imagedir, 'rowslice.png')
        rowslice = np.array(Image.open(rowslice))
        rowslicecp = rowslice.copy()
        rowslicehalf = rowslicecp.shape[0] // 2
        rowslicecp[rowslicehalf:rowslicehalf+3, :] = 255
        addedimg = marker.addPointSlice2Image(demot,
                                              point,
                                              beforeAfterCoord=(before, after),
                                              colSlice=colSlice,
                                              imgSlice=rowslicecp,
                                              isUpTo=isUpTo)
        compimg = os.path.join(self.imagedir, 'rowAddedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(addedimg, compimg,
                           "Added row does not give added image"
                           )

    def test_seammarker_markSeam4Point_colSliceTrue_isUpToFalse(self):
        isUpTo = False
        colSlice = True
        viet = self.loadImageCol()
        points = loadJfile(self.points_down_path)
        point = points["0"]
        point_coord = (point['y'], point['x'])
        marker = SeamMarker(viet)
        markedImage = marker.markSeam4Point(viet,
                                            point_coord,
                                            isUpTo,
                                            point['threshold'],
                                            colSlice,
                                            mark_color=[0, 255, 0]
                                            )
        compimg = os.path.join(self.imagedir, 'colMarkedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(markedImage, compimg,
                           "Marked image is not equivalent to expected image")

    def test_seammarker_markSeam4Point_colSliceFalse_isUpToTrue(self):
        isUpTo = True
        colSlice = False
        demot = self.loadImageRow()
        points = loadJfile(self.points_left_path)
        point = points["2"]
        point_coord = (point['y'], point['x'])
        marker = SeamMarker(demot)
        markedImage = marker.markSeam4Point(demot,
                                            point_coord,
                                            isUpTo,
                                            point['threshold'],
                                            colSlice,
                                            mark_color=[0, 255, 0]
                                            )
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'rowMarkedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(markedImage, compimg,
                           "Marked image is not equivalent to expected image")

    def test_seammarker_markPointSeam_down(self):
        viet = self.loadImageCol()
        points = loadJfile(self.points_down_path)
        point = points["0"]
        point_coord = (point['y'], point['x'])
        marker = SeamMarker(viet)
        markedImage = marker.markPointSeam(img=viet.copy(),
                                           point=point_coord,
                                           direction=point['direction'],
                                           thresh=point['threshold']
                                           )
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'colMarkedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(markedImage, compimg,
                           "Marked image is not equivalent to expected image")

    def test_seammarker_markPointSeam_left(self):
        demot = self.loadImageRow()
        points = loadJfile(self.points_left_path)
        point = points["2"]
        point_coord = (point['y'], point['x'])
        marker = SeamMarker(demot)
        markedImage = marker.markPointSeam(img=demot,
                                           point=point_coord,
                                           direction=point['direction'],
                                           thresh=point['threshold']
                                           )
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'rowMarkedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(markedImage, compimg,
                           "Marked image is not equivalent to expected image")

    def test_seammarker_markPointListSeam_down(self):
        viet = self.loadImageCol()
        marker = SeamMarker(viet)
        points = loadJfile(self.points_down_path)
        markedImage = marker.markPointListSeam(img=viet,
                                               plist=points)
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'markedColumnsImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(
            markedImage, compimg,
            "Marked image is not equivalent to expected image"
        )

    def test_seammarker_markPointListSeam_left(self):
        demot = self.loadImageRow()
        marker = SeamMarker(demot)
        points = loadJfile(self.points_left_path)
        markedImage = marker.markPointListSeam(img=demot,
                                               plist=points)
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'markedRowsImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(
            markedImage, compimg,
            "Marked image is not equivalent to expected image"
        )

    def test_seammarker_matchMarkCoordPairLength_down_colSliceTrue(self):
        colSlice = True
        isUpTo = False
        pair = getCoordPair(self.coords_down_path, colSlice)
        coords = getCoordsDict(self.coords_down_path)
        point1 = pair[0]
        point2 = pair[1]
        marker = SeamMarker(img=np.zeros((2, 2), dtype=np.uint8))
        coord1 = coords[point1]
        coord2 = coords[point2]
        retval1, retval2 = marker.matchMarkCoordPairLength(
            coord1, coord2, colSlice, isUpTo)
        # pdb.set_trace()
        self.assertEqual(retval1[-1][0], retval2[-1][0])

    def test_seammarker_matchMarkCoordPairLength_left_colSliceFalse(self):
        colSlice = False
        isUpTo = True
        pair = getCoordPair(self.coords_left_path, colSlice)
        coords = getCoordsDict(self.coords_left_path)
        point1 = pair[0]
        point2 = pair[1]
        marker = SeamMarker(img=np.zeros((2, 2), dtype=np.uint8))
        coord1 = coords[point1]
        coord2 = coords[point2]
        # pdb.set_trace()
        retval1, retval2 = marker.matchMarkCoordPairLength(
            coord1, coord2, colSlice, isUpTo)
        self.assertEqual(retval1[-1][1], retval2[-1][1])

    def test_seammarker_swapAndSliceMarkCoordPair_down_colSliceTrue(self):
        viet = self.loadImageCol()
        # pdb.set_trace()
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
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'slicedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(
            compimg, swaped, "Image slicing with mark coordinate has failed"
        )

    def test_seammarker_sliceImageWithMarkCoordPair_down_colSliceTrue(self):
        "test seam marker slice Image with mark coordinate pair"
        viet = self.loadImageCol()
        marker = SeamMarker(img=np.zeros((2, 2), dtype=np.uint8))
        coords = getCoordsDict(self.coords_down_path)
        colSlice = True
        isUpTo = False
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
                                                     colSlice, isUpTo)
        # pdb.set_trace()
        compimg = os.path.join(self.imagedir, 'slicedImage.png')
        compimg = np.array(Image.open(compimg))
        self.compareArrays(compimg,
                           segment,
                           "Image slicing with mark coordinate has failed")

    def test_seammarker_segmentImageWithPointListSeamCoordinate(self):
        img = self.loadImageCol()
        marker = SeamMarker(img=np.zeros((2, 2), dtype=np.uint8))
        coords = prepPointCoord(self.coords_down_path)
        segments = marker.segmentImageWithPointListSeamCoordinate(
            image=img, coords=coords)
        compsegs = self.loadSegments()
        message = "Segment {0} failed"
        # pdb.set_trace()
        compvals = [
            [self.compareArrays(
                segs[i],
                compsegs[i],
                message.format(str(i))
            ) for i in range(len(segs))
            ] for direction, segs in segments.items()
        ]

    def test_seammarker_getPointSeamCoordinate_down(self):
        viet = self.loadImageCol()
        points = getPointListFromPointPath(self.points_down_path)
        point = tuple(points[0])
        marker = SeamMarker(viet)
        jfile = loadJfile(self.coords_down_path)
        # pdb.set_trace()
        pointData = getCoordWithPoint(self.coords_down_path, point)
        thresh_val = pointData['threshold']
        direction = pointData['direction']
        coords = marker.getPointSeamCoordinate(viet.copy(),
                                               point,
                                               direction=direction,
                                               thresh=thresh_val)
        coords = shapeCoordinate(coords)

        ccoord = np.array(pointData["seamCoordinates"], dtype=np.int)
        # pdb.set_trace()
        self.compareArrays(coords, ccoord,
                           "Computed coordinates are not equal")


if __name__ == "__main__":
    unittest.main()
