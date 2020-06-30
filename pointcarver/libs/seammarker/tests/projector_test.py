# author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE
# test projector.py

from seammarker import projector as pj

import unittest
import os
from PIL import Image
import numpy as np
# import json
# import pdb


class ProjectorTest(unittest.TestCase):
    "Test projector"

    def setUp(self):
        "Set up test files"
        currentdir = str(__file__)
        currentdir = os.path.join(currentdir, os.pardir)
        currentdir = os.path.join(currentdir, os.pardir)
        currentdir = os.path.abspath(currentdir)
        testdir = os.path.join(currentdir, "tests")
        assetdir = os.path.join(testdir, 'assets')
        self.assetdir = assetdir
        self.imagedir = os.path.join(assetdir, 'images')
        self.utilsImageDir = os.path.join(self.imagedir, 'utils')
        self.projectorImageDir = os.path.join(self.imagedir, 'projector')
        jsondir = os.path.join(assetdir, 'jsonfiles')
        self.npdir = os.path.join(assetdir, 'numpyfiles')
        self.projectorNumpyDir = os.path.join(self.npdir, 'projector')
        self.image_col_path = os.path.join(self.imagedir, 'vietHard.jpg')

        # checks
        self.GEN_SLICE_IMAGE_WITH_MASK = False
        self.slice_img_path = os.path.join(self.projectorImageDir,
                                           "sliceImageWithMask.png")
        self.GEN_SLICE_ELLIPSE_IMAGE_MASK = False
        self.slice_img_ellipse_path = os.path.join(self.projectorImageDir,
                                                   "sliceEllipseFromImage.png")
        self.GEN_SLICE_THICKLINE_IMAGE_MASK = False
        self.slice_img_thickline_path = os.path.join(
            self.projectorImageDir,
            "sliceThickLineFromImage.png"
        )
        self.GEN_SLICE_POLYGON_IMAGE = False
        self.slice_img_polygon_path = os.path.join(self.projectorImageDir,
                                                   "slicePolygonFromImage.png")
        self.GEN_SLICE_ELLIPSE_ARRAY = False
        self.slice_arr_ellipse_path = os.path.join(self.projectorNumpyDir,
                                                   "sliceEllipseFromImage.npy")
        self.GEN_SLICE_THICKLINE_ARRAY = False
        self.slice_arr_thickline_path = os.path.join(
            self.projectorNumpyDir,
            "sliceThickLineFromImage.npy"
        )
        self.GEN_SLICE_POLYGON_ARRAY = False
        self.slice_arr_polygon_path = os.path.join(self.projectorNumpyDir,
                                                   "slicePolygonFromImage.npy")
        self.GEN_ELLIPSE_HALVES_IMAGE = False
        self.ellipse_half1_img_path = os.path.join(self.projectorImageDir,
                                                   "cutEllipse2Half1.png")
        self.ellipse_half2_img_path = os.path.join(self.projectorImageDir,
                                                   "cutEllipse2Half2.png")
        self.GEN_ELLIPSE_HALVES_ARR = False
        self.ellipse_half1_arr_path = os.path.join(self.projectorNumpyDir,
                                                   "cutEllipse2Half1Mask.npy")
        self.ellipse_half2_arr_path = os.path.join(self.projectorNumpyDir,
                                                   "cutEllipse2Half2Mask.npy")
        self.syntetic_half_img_path = os.path.join(self.projectorImageDir,
                                                   "synteticFirstHalf.png")
        self.syntetic_appended_img_path = os.path.join(self.projectorImageDir,
                                                       "synteticAppended.png")
        self.syntetic_half_arr_path = os.path.join(self.projectorNumpyDir,
                                                   "synteticFirstMask.npy")
        self.syntetic_appended_arr_path = os.path.join(
            self.projectorNumpyDir,
            "synteticAppendedMask.npy")
        self.syntetic_orig_arr_path = os.path.join(self.projectorNumpyDir,
                                                   "origImageMask.npy")

    def compareArrays(self, arr1, arr2, message):
        "Compare arrays for equality"
        result = arr1 == arr2
        result = result.all()
        self.assertTrue(result, message)

    def generateSliceImageWithMask(self):
        if self.GEN_SLICE_IMAGE_WITH_MASK:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 20
            height = 30
            halfrow = rownb // 2
            halfcol = colnb // 2
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            mask = np.zeros((rownb, colnb), dtype=np.bool)
            mask[y1:y2+1, x1:x2+1] = True
            newimg = pj.sliceImageWithMask(vietImg.copy(), mask)
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_path)

    def generateSliceEllipseFromImageWithoutCoordinate(self):
        if self.GEN_SLICE_ELLIPSE_IMAGE_MASK:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 200
            height = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            newimg = pj.sliceEllipseFromImage(vietImg.copy(),
                                              bbox={"x1": x1,
                                                    "x2": x2,
                                                    "y1": y1,
                                                    "y2": y2},
                                              withCoord=False)
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_ellipse_path)

    def generateSliceThickLineFromImageWithoutCoordinate(self):
        if self.GEN_SLICE_THICKLINE_IMAGE_MASK:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 200
            height = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            thickness = 6
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            newimg = pj.sliceThickLineFromImage(vietImg.copy(),
                                                bbox={"x1": x1,
                                                      "x2": x2,
                                                      "y1": y1,
                                                      "y2": y2},
                                                line_width=thickness,
                                                withCoord=False)
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_thickline_path)

    def generateSlicePolygonFromImageWithoutCoordinate(self):
        if self.GEN_SLICE_POLYGON_IMAGE:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width1 = 200
            height1 = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            p1 = {"x": halfcol - width1, "y": halfrow - height1}
            p2 = {"x": p1['x'] + width1 // 2, "y": p1['y']}
            p3 = {"x": halfcol, "y": halfrow}
            p4 = {"x": halfcol + width1, "y": halfrow + height1}
            p5 = {"x": halfcol + width1 // 3, "y": halfrow + height1 // 2}
            coords = [p1, p2, p3, p4, p5]
            newimg = pj.slicePolygonFromImage(vietImg.copy(), coords,
                                              withCoord=False)
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_polygon_path)

    def generateSliceEllipseFromImageWithCoordinate(self):
        if self.GEN_SLICE_ELLIPSE_ARRAY:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 200
            height = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            newimg, coords = pj.sliceEllipseFromImage(vietImg.copy(),
                                                      bbox={"x1": x1,
                                                            "x2": x2,
                                                            "y1": y1,
                                                            "y2": y2},
                                                      withCoord=True)
            np.save(self.slice_arr_ellipse_path, coords)

    def generateSliceThickLineFromImageWithCoordinate(self):
        if self.GEN_SLICE_THICKLINE_ARRAY:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 200
            height = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            thickness = 6
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            newimg, coords = pj.sliceThickLineFromImage(vietImg.copy(),
                                                        bbox={"x1": x1,
                                                              "x2": x2,
                                                              "y1": y1,
                                                              "y2": y2},
                                                        line_width=thickness,
                                                        withCoord=True)
            np.save(self.slice_arr_thickline_path, coords)

    def generateSlicePolygonFromImageWithCoordinate(self):
        if self.GEN_SLICE_POLYGON_ARRAY:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width1 = 200
            height1 = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            p1 = {"x": halfcol - width1, "y": halfrow - height1}
            p2 = {"x": p1['x'] + width1 // 2, "y": p1['y']}
            p3 = {"x": halfcol, "y": halfrow}
            p4 = {"x": halfcol + width1, "y": halfrow + height1}
            p5 = {"x": halfcol + width1 // 3, "y": halfrow + height1 // 2}
            coords = [p1, p2, p3, p4, p5]
            newimg, coords = pj.slicePolygonFromImage(vietImg.copy(), coords,
                                                      withCoord=True)
            np.save(self.slice_arr_polygon_path, coords)

    def test_sliceImageWithMask(self):
        "test slice image with mask works for a rectangle"
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 20
        height = 30
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        mask = np.zeros_like(vietImg, dtype=np.bool)
        mask[y1:y2+1, x1:x2+1] = True
        newimg = pj.sliceImageWithMask(vietImg.copy(), mask)
        compimg = Image.open(self.slice_img_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_sliceEllipseFromImageWithoutCoord(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        withCoord = False
        newimg = pj.sliceEllipseFromImage(vietImg.copy(),
                                          bbox={"x1": x1,
                                                "x2": x2,
                                                "y1": y1,
                                                "y2": y2},
                                          withCoord=withCoord)
        compimg = Image.open(self.slice_img_ellipse_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_sliceThickLineFromImageWithoutCoord(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        thickness = 6
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        withCoord = False
        newimg = pj.sliceThickLineFromImage(vietImg.copy(),
                                            bbox={"x1": x1,
                                                  "x2": x2,
                                                  "y1": y1,
                                                  "y2": y2},
                                            line_width=thickness,
                                            withCoord=withCoord)
        compimg = Image.open(self.slice_img_thickline_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_slicePolygonFromImageWithoutCoord(self):
        "Test if we can slice a polygon from image"
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width1 = 200
        height1 = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        p1 = {"x": halfcol - width1, "y": halfrow - height1}
        p2 = {"x": p1['x'] + width1 // 2, "y": p1['y']}
        p3 = {"x": halfcol, "y": halfrow}
        p4 = {"x": halfcol + width1, "y": halfrow + height1}
        p5 = {"x": halfcol + width1 // 3, "y": halfrow + height1 // 2}
        coords = [p1, p2, p3, p4, p5]
        withCoord = False
        newimg = pj.slicePolygonFromImage(vietImg.copy(),
                                          coords, withCoord=withCoord)
        compimg = Image.open(self.slice_img_polygon_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_sliceEllipseFromImageWithCoord(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        newimg, coords = pj.sliceEllipseFromImage(vietImg.copy(),
                                                  bbox={"x1": x1,
                                                        "x2": x2,
                                                        "y1": y1,
                                                        "y2": y2},
                                                  withCoord=True)
        comparr = np.load(self.slice_arr_ellipse_path)
        self.compareArrays(comparr, coords,
                           "Image Slice not same with expected image")

    def test_sliceThickLineFromImageWithCoord(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        thickness = 6
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        withCoord = True
        newimg, coords = pj.sliceThickLineFromImage(vietImg.copy(),
                                                    bbox={"x1": x1,
                                                          "x2": x2,
                                                          "y1": y1,
                                                          "y2": y2},
                                                    line_width=thickness,
                                                    withCoord=withCoord)
        comparr = np.load(self.slice_arr_thickline_path)
        self.compareArrays(comparr, coords,
                           "Image Slice not same with expected image")

    def test_slicePolygonFromImageWithCoord(self):
        "Test if we can slice a polygon from image"
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width1 = 200
        height1 = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        p1 = {"x": halfcol - width1, "y": halfrow - height1}
        p2 = {"x": p1['x'] + width1 // 2, "y": p1['y']}
        p3 = {"x": halfcol, "y": halfrow}
        p4 = {"x": halfcol + width1, "y": halfrow + height1}
        p5 = {"x": halfcol + width1 // 3, "y": halfrow + height1 // 2}
        coords = [p1, p2, p3, p4, p5]
        withCoord = True
        newimg, coords = pj.slicePolygonFromImage(vietImg.copy(),
                                                  coords, withCoord=withCoord)
        comparr = np.load(self.slice_arr_polygon_path)
        self.compareArrays(comparr, coords,
                           "Image Slice not same with expected image")

    def test_cutEllipse2HalfWithoutMask(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        firstHalf, secondHalf = pj.cutEllipse2Half(vietImg.copy(),
                                                   bbox={"x1": x1,
                                                         "x2": x2,
                                                         "y1": y1,
                                                         "y2": y2},
                                                   withMask=False)
        compimg1 = Image.open(self.ellipse_half1_img_path)
        compimg2 = Image.open(self.ellipse_half2_img_path)
        comparr1, comparr2 = np.array(compimg1), np.array(compimg2)
        self.compareArrays(firstHalf, compimg1,
                           "First half of ellipse not correctly cut")
        self.compareArrays(secondHalf, compimg2,
                           "Second half of ellipse not correctly cut")

    def test_cutEllipse2HalfWithMask(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        (firstHalf, mask1,
         secondHalf, mask2) = pj.cutEllipse2Half(vietImg.copy(),
                                                 bbox={"x1": x1,
                                                       "x2": x2,
                                                       "y1": y1,
                                                       "y2": y2},
                                                 withMask=True)
        compimg1 = Image.open(self.ellipse_half1_img_path)
        compimg2 = Image.open(self.ellipse_half2_img_path)
        compmask1 = np.load(self.ellipse_half1_arr_path)
        compmask2 = np.load(self.ellipse_half2_arr_path)
        comparr1, comparr2 = np.array(compimg1), np.array(compimg2)
        self.compareArrays(firstHalf, compimg1,
                           "First half of ellipse not correctly cut")
        self.compareArrays(secondHalf, compimg2,
                           "Second half of ellipse not correctly cut")
        self.compareArrays(mask1, compmask1,
                           "First half mask not correct")
        self.compareArrays(mask2, compmask2,
                           "Second half mask not correct")

    def test_findCenterPoint(self):
        bbox = {"x1": 10, "x2": 20, "y1": 30, "y2": 40}
        compCenter = {"x": 15, "y": 35}
        center = pj.findCenterInBbox(bbox)
        self.assertDictEqual(compCenter, center)

    def test_appendSecondHalf2FirstHalfEllipseWithoutMask(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        bbox = {"x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2}
        center = {"x": (bbox["x1"] + bbox["x2"]) // 2,
                  "y": (bbox["y1"] + bbox["y2"]) // 2}
        firstHalf = np.array(Image.open(self.ellipse_half1_img_path).copy())
        secondHalf = np.array(Image.open(self.ellipse_half2_img_path).copy())
        firstMask = np.load(self.ellipse_half1_arr_path)
        secondMask = np.load(self.ellipse_half2_arr_path)
        (synteticFirstHalf, synteticAppended,
         voffset, hoffset, (rownb,
                            colnb, pixnb)) = pj.appendSecondHalf2FirstHalfEllipse(
            firstHalf=firstHalf,
            firstHalfMask=firstMask,
            secondHalf=secondHalf,
            secondHalfMask=secondMask,
            centerPoint=center,
            bbox=bbox,
            shapePixelValue=255,
            withMask=False)
        comprownb, comcolnb, compixnb = firstHalf.shape
        compvoffset = max(bbox['y1'], bbox['y2']) - min(bbox['y1'],
                                                        bbox['y2'])
        # comphoffset = center['x'] - min(bbox['x1'], bbox['x2'])
        comphoffset = 0
        comparr1 = np.array(Image.open(self.syntetic_half_img_path))
        comparr2 = np.array(Image.open(self.syntetic_appended_img_path))
        self.assertEqual(compvoffset, voffset)
        self.assertEqual(comphoffset, hoffset)
        self.assertEqual((rownb, colnb, pixnb),
                         (comprownb, comcolnb, compixnb))
        self.compareArrays(comparr1, synteticFirstHalf, "First halves")
        self.compareArrays(comparr2, synteticAppended, "Appended syntetic")

    def test_appendSecondHalf2FirstHalfEllipseWithMask(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        bbox = {"x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2}
        center = {"x": (bbox["x1"] + bbox["x2"]) // 2,
                  "y": (bbox["y1"] + bbox["y2"]) // 2}
        firstHalf = np.array(Image.open(self.ellipse_half1_img_path).copy())
        secondHalf = np.array(Image.open(self.ellipse_half2_img_path).copy())
        firstMask = np.load(self.ellipse_half1_arr_path)
        secondMask = np.load(self.ellipse_half2_arr_path)
        (synteticFirstHalf, synteticFirstMask,
         synteticAppended, synteticAppendedMask,
         voffset, hoffset, (rownb,
                            colnb,
                            pixnb)) = pj.appendSecondHalf2FirstHalfEllipse(
            firstHalf=firstHalf,
            firstHalfMask=firstMask,
            secondHalf=secondHalf,
            secondHalfMask=secondMask,
            centerPoint=center,
            bbox=bbox,
            shapePixelValue=255,
            withMask=True)
        comprownb, comcolnb, compixnb = firstHalf.shape
        compvoffset = max(bbox['y1'], bbox['y2']) - min(bbox['y1'],
                                                        bbox['y2'])
        # comphoffset = center['x'] - min(bbox['x1'], bbox['x2'])
        comphoffset = 0
        comparr1 = np.array(Image.open(self.syntetic_half_img_path))
        comparr2 = np.array(Image.open(self.syntetic_appended_img_path))
        # pdb.set_trace()
        comparr3 = np.load(self.syntetic_half_arr_path)
        comparr4 = np.load(self.syntetic_appended_arr_path)
        self.assertEqual(compvoffset, voffset)
        self.assertEqual(comphoffset, hoffset)
        self.assertEqual((rownb, colnb, pixnb),
                         (comprownb, comcolnb, compixnb))
        self.compareArrays(comparr1, synteticFirstHalf, "First halves")
        self.compareArrays(comparr2, synteticAppended, "Appended syntetic")
        self.compareArrays(comparr3, synteticFirstMask, "First halves")
        self.compareArrays(comparr4, synteticAppendedMask, "First halves")

    def test_putsecondHalf2OriginalPosition_withoutMask(self):
        appendedImg = np.array(Image.open(self.syntetic_appended_img_path))
        firstHalfMask = np.load(self.syntetic_half_arr_path)
        secondHalfMask = np.load(self.ellipse_half2_arr_path)
        appendedMask = np.load(self.syntetic_appended_arr_path)
        vietImg = np.array(Image.open(self.image_col_path).copy())
        imshape = vietImg.shape
        origImg = pj.putSecondHalf2OriginalPosition(
            appendedSynteticImage=appendedImg,
            firstHalfMask=firstHalfMask,
            secondHalfMask=secondHalfMask,
            appendedMask=appendedMask,
            imshape=imshape,
            shapePixelValue=255,
            withMask=False)
        compimg = np.array(Image.open(self.slice_img_ellipse_path))
        self.compareArrays(origImg,
                           compimg, "putback image not same with original")

    def test_putsecondHalf2OriginalPosition_withMask(self):
        appendedImg = np.array(Image.open(self.syntetic_appended_img_path))
        firstHalfMask = np.load(self.syntetic_half_arr_path)
        secondHalfMask = np.load(self.ellipse_half2_arr_path)
        appendedMask = np.load(self.syntetic_appended_arr_path)
        vietImg = np.array(Image.open(self.image_col_path).copy())
        imshape = vietImg.shape
        (origImg, origImgMask) = pj.putSecondHalf2OriginalPosition(
            appendedSynteticImage=appendedImg,
            secondHalfMask=secondHalfMask,
            firstHalfMask=firstHalfMask,
            appendedMask=appendedMask,
            imshape=imshape,
            shapePixelValue=255,
            withMask=True)
        compimg = np.array(Image.open(self.slice_img_ellipse_path))
        comparr = np.load(self.syntetic_orig_arr_path)
        self.compareArrays(origImg,
                           compimg, "putback image not same with original")
        self.compareArrays(origImgMask,
                           comparr, "putback image not same with original")


if __name__ == "__main__":
    unittest.main()
