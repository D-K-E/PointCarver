# mark seams in texts
# author: Kaan Eraslan
# Implementation in part taken from the link:
# https://karthikkaranth.me/blog/implementing-seam-carving-with-python/):

# Author in the link: Karthik Karanth

import numpy as np  # array/matrix manipulation
import scipy.ndimage as nd  # operate easily on image matrices
import pdb


class SeamFuncs:
    def __init__(self):
        self.mark_color = [0, 255, 0]

    def calc_energy(self,
                    img: np.ndarray([], dtype=np.uint8)):
        filter_du = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        filter_du = np.stack([filter_du] * 3, axis=2)

        filter_dv = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        filter_dv = np.stack([filter_dv] * 3, axis=2)

        img = img.astype('float32')
        convolved = np.absolute(nd.filters.convolve(
            img, filter_du)) + np.absolute(
                nd.filters.convolve(img, filter_dv))

        # We sum the energies in the red, green, and blue channels
        energy_map = convolved.sum(axis=2)

        return energy_map

    def minimum_seam(self, img: np.ndarray([], dtype=np.uint8),
                     emap=None):
        r, c, _ = img.shape

        # if the energy map is already calculated
        if emap is not None:
            energy_map = emap
        else:
            energy_map = self.calc_energy(img)

        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=np.int)

        for i in range(1, r):
            for j in range(0, c):
                # Handle the left edge of the image, to ensure we don't index -1
                if j == 0:
                    maprow = M[i - 1, j:j + 2]
                    idx = np.argmin(maprow)
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]

                M[i, j] += min_energy

        return M, backtrack

    def mark_column(self,
                    img: np.ndarray([], dtype=np.uint8),
                    emap=None,
                    mark_color=[250, 120, 120]  # yellow
                    ):
        r, c, _ = img.shape
        imcopy = img.copy()

        M, backtrack = self.minimum_seam(img, emap)

        # Create a (r, c) matrix filled with the value True
        # We'll be marking all pixels from the image which
        # have False later
        mask = np.zeros((r, c), dtype=np.bool)

        # Find the position of the smallest element in the
        # last row of M
        j = np.argmin(M[-1])

        for i in reversed(range(r)):
            # Mark the pixels
            # and save mark positions for later use
            mask[i, j] = True
            j = backtrack[i, j]

        # Since the image has 3 channels, we convert our
        # mask to 3D
        mask = np.stack([mask] * 3, axis=2)

        # mark the pixels with the given mark color
        imcopy = np.where(mask, mark_color, img)

        return imcopy, mask

    def mark_row(self,
                 img: np.ndarray([], dtype=np.uint8),
                 mark_color=[250, 120, 120]):
        img = np.rot90(img, 1, (0, 1))
        img, mask = self.mark_column(img, mark_color=mark_color)
        img = np.rot90(img, 3, (0, 1))
        mask = np.rot90(mask, 3, (0, 1))
        return img, mask


class SeamMarker(SeamFuncs):
    def __init__(self,
                 img: np.ndarray([], dtype=np.uint8),
                 plist=[],
                 thresh=10,
                 direction='down'):
        super().__init__()
        self.img = img
        self.plist = plist
        self.direction = direction
        self.thresh = thresh
        self.mark_color = [0, 255, 0]

    def expandPointCoordinate(self,
                              ubound: int,
                              coord: int,
                              thresh: int):
        "Expand the coordinate with ubound using given threshold"
        assert thresh <= 100 and thresh > 0
        assert isinstance(thresh, int) is True
        sliceAmount = ubound * thresh // 100
        sliceHalf = sliceAmount // 2
        coordBefore = coord - sliceHalf
        coordAfter = coord + sliceHalf
        if coordBefore < 0:
            coordBefore = 0
        if coordAfter >= ubound:
            coordAfter = ubound - 1
        return coordBefore, coordAfter

    def getColumnSliceOnPoint(self,
                              point: (int, int),
                              img: np.ndarray,
                              isUpTo: bool,
                              thresh: int):
        "Get column slice on point"
        col_nb = img.shape[1]
        pointCol = point[1]
        colBefore, colAfter = self.expandPointCoordinate(ubound=col_nb,
                                                         coord=pointCol,
                                                         thresh=thresh)
        if isUpTo is False:
            imgSlice = img[point[0]:, colBefore:colAfter]
        else:
            imgSlice = img[:point[0], colBefore:colAfter]
        #
        return imgSlice, (colBefore, colAfter)

    def getRowSliceOnPoint(self,
                           point: (int, int),
                           img: np.ndarray,
                           isUpTo: bool,
                           thresh: int):
        "Get row slice on point"
        row_nb = img.shape[0]
        pointRow = point[0]
        rowBefore, rowAfter = self.expandPointCoordinate(ubound=row_nb,
                                                         coord=pointRow,
                                                         thresh=thresh)
        if isUpTo is False:
            imgSlice = img[rowBefore:rowAfter, point[1]:]
        else:
            imgSlice = img[rowBefore:rowAfter, :point[1]]
        #
        return imgSlice, (rowBefore, rowAfter)

    def sliceOnPoint(self, img: np.ndarray([], dtype=np.uint8),
                     point: (int, int),
                     isUpTo=False,
                     colSlice=False,
                     thresh=3) -> np.ndarray([], dtype=np.uint8):
        """
        Slice based on the point with given threshold as percent

        Description
        ------------
        We take the point as a center. Then we calculate
        the amount of slicing based on the threshold.
        Threshold is a percent value. Thus the sliced amount
        is relative to the image shape.
        Once the slicing amount is computed, we slice the given
        amount from the image. The point should be at the center
        of a side of the sliced area.

        Parameters
        ------------
        img: np.ndarray([], dtype=np.uint8)

        point: (int, int)
            coordinate of row, column for the point


        isUpTo: boolean
            determines whether we slice up to the point or from point
            onwards

        thresh: int
            threshold value for the amount of slicing. It should be
            between 0 - 100.

        Return
        -------

        imgSlice: np.ndarray(dtype=np.uint8)
        """
        # make sure threshold is a percent and an integer
        assert thresh <= 100 and thresh > 0
        assert isinstance(thresh, int) is True

        if colSlice is True:
            imgSlice, (before, after) = self.getColumnSliceOnPoint(point,
                                                                   img,
                                                                   isUpTo,
                                                                   thresh)
        else:
            imgSlice, (before, after) = self.getRowSliceOnPoint(point,
                                                                img,
                                                                isUpTo,
                                                                thresh)

        return imgSlice, (before, after)

    def addColumnSlice2Image(self, image: np.ndarray,
                             point: (int, int), beforeAfterCoord: (int, int),
                             imgSlice: np.ndarray, isUpTo: bool):
        "Add column slice 2 image"
        imcp = image.copy()
        before, after = beforeAfterCoord
        if isUpTo is False:
            imcp[point[0]:, before:after] = imgSlice
        else:
            imcp[:point[0], before:after] = imgSlice
        return imcp

    def addRowSlice2Image(self, image: np.ndarray,
                          point: (int, int), beforeAfterCoord: (int, int),
                          imgSlice: np.ndarray, isUpTo: bool):
        "Row slice 2 image"
        imcp = image.copy()
        before, after = beforeAfterCoord
        if isUpTo is False:
            imcp[before:after, point[1]:] = imgSlice
        else:
            imcp[before:after, :point[1]] = imgSlice
        return imcp

    def addPointSlice2Image(self,
                            img: np.ndarray([], dtype=np.uint8),
                            point: (int, int),  # y, x
                            beforeAfterCoord: (int, int),
                            colSlice: bool,
                            imgSlice: np.ndarray([], dtype=np.uint8),
                            isUpTo: bool,
                            ):
        "Add sliced zone back to image"
        # pdb.set_trace()
        if colSlice is True:
            imcp = self.addColumnSlice2Image(image=img,
                                             point=point,
                                             beforeAfterCoord=beforeAfterCoord,
                                             imgSlice=imgSlice,
                                             isUpTo=isUpTo)
        else:
            imcp = self.addRowSlice2Image(image=img,
                                          point=point,
                                          beforeAfterCoord=beforeAfterCoord,
                                          imgSlice=imgSlice,
                                          isUpTo=isUpTo)
        #
        return imcp

    def getMarkCoordinates(self,
                           markedMask: np.ndarray([], dtype=np.bool)):
        "Get marked coordinates from the mask"
        indexArray = markedMask.nonzero()
        indexArray = np.array(indexArray)
        indexArray = indexArray.T
        # indexArray[0] == [rowPosition, colPosition, colorPosition]
        return indexArray

    def sortCoords4Matching(self, coord1, coord2,
                            colSlice: bool):
        "Sort coordinates with respect to column slicing"
        if colSlice is True:
            # sort by y val
            ind1 = np.argsort(coord1[:, 0])
            coord1 = coord1[ind1]
            ind2 = np.argsort(coord2[:, 0])
            coord2 = coord2[ind2]
        elif colSlice is False:
            # sort by x val
            ind1 = np.argsort(coord1[:, 1])
            coord1 = coord1[ind1]
            ind2 = np.argsort(coord2[:, 1])
            coord2 = coord2[ind2]
        return coord1, coord2

    def getKeepAndLimitValues(self, coord1,
                              coord2, colSlice: bool,
                              isUpTo: bool):
        "Get keep value and limit values to generate fill values later on"
        if isUpTo is False:
            axval = 0
        else:
            axval = -1
        if colSlice is True:
            keepval = 1
            rangeval = 0
        else:
            keepval = 0
            rangeval = 1
        #
        COORD1KEEP = coord1[axval, keepval]
        COORD2KEEP = coord2[axval, keepval]
        coord1val = coord1[axval, rangeval]
        coord2val = coord2[axval, rangeval]
        if isUpTo is False:
            # so their first values should match
            # make sure coord1 is the one with smaller value
            # we'll prepend values to coord2 later on
            if coord1val >= coord2val:
                coord1, coord2 = coord2, coord1
                coord1val, coord2val = coord2val, coord1val
                COORD1KEEP, COORD2KEEP = COORD2KEEP, COORD1KEEP
        else:
            # their last values should match
            # make sure coord2 is the one with smaller value
            # making it shorter
            # we'll append values to coord2 later on
            if coord2val >= coord1val:
                coord1, coord2 = coord2, coord1
                coord1val, coord2val = coord2val, coord1val
                COORD1KEEP, COORD2KEEP = COORD2KEEP, COORD1KEEP
        #
        return (coord1, coord2,
                COORD1KEEP, COORD2KEEP,
                coord1val, coord2val)

    def prepCoords2Matching(self, coord1, coord2,
                            colSlice: bool,
                            isUpTo: bool):
        "Prepare coordinates to matching"
        assert isinstance(colSlice, bool)
        coord1, coord2 = self.sortCoords4Matching(coord1,
                                                  coord2,
                                                  colSlice)
        assert isinstance(isUpTo, bool)
        (coord1, coord2,
         COORD1KEEP, COORD2KEEP,
         coord1val, coord2val) = self.getKeepAndLimitValues(coord1,
                                                            coord2, colSlice,
                                                            isUpTo)
        if isUpTo is False:
            fillvals = [i for i in range(coord2val-1, # since we prepend
                                         # this array later on
                                         coord1val-1, -1)]
        else:
            fillvals = [i for i in range(coord2val+1, coord1val+1, 1)]
        #
        return (coord1, coord2, 
                COORD1KEEP, COORD2KEEP, 
                coord1val, coord2val,
                fillvals)

    def matchMarkCoordPairLength(self, coord1, coord2,
                                 colSlice: bool,
                                 isUpTo: bool):
        """Match mark coordinate pairs

        Purpose
        ---------

        Matches the coordinate pairs that start from different points.

        Description
        ------------

        The logic is simple. If we are dealing with a column slice, then
        the y values should match, since we need to have equal column length
        to fill the column mask later on. That's why we sort the coordinates
        by y value at first, to make sure that their top points are closest
        to each other.

        If we are dealing with a row slice, then the x values match since
        we need to have equal line length to fill the line mask later on

        We simply keep the last first value of the unmatched axis of
        the coordinate array. You can think of matching axes as drawing a
        parallel line from the point where a coordinate array falls short,
        up until it matches the other coordinate array's limit.

        """
        assert coord1.shape[1] == 2 and coord2.shape[1] == 2
        # col slice determines the axis of match
        (coord1, coord2,
         COORD1KEEP, COORD2KEEP,
         coord1val, coord2val,
         fillvals) = self.prepCoords2Matching(
             coord1, coord2, colSlice, isUpTo)
        for i in fillvals:
            if colSlice is True:  # column slice
                if isUpTo is False:
                    coord2 = np.insert(coord2, 0, [i, COORD2KEEP], axis=0)
                else:
                    coord2 = np.insert(coord2, coord2.shape[0],
                                       [i, COORD2KEEP], axis=0)
            else:
                if isUpTo is False:
                    coord2 = np.insert(coord2, 0, [COORD2KEEP, i], axis=0)
                else:
                    coord2 = np.insert(coord2, coord2.shape[0],
                                       [COORD2KEEP, i], axis=0)
        return coord1, coord2

    def swapAndSliceMarkCoordPair(self, markCoord1, markCoord2,
                                  image, colSlice: bool) -> np.ndarray:
        "Slice image using mark coordinate pair"
        imcp = np.copy(image)
        mask = np.zeros_like(imcp)
        if colSlice is True:
            # then comparing x values
            fsum = np.sum(markCoord1[:, 1] - markCoord2[:, 1], dtype=np.int)
            if fsum >= 0:
                markCoord1, markCoord2 = markCoord2, markCoord1
            #
            # pdb.set_trace()
            for i in range(markCoord1.shape[0]):
                startx = markCoord1[i, 1]
                yval = markCoord1[i, 0]
                endx = markCoord2[i, 1]
                mask[yval, startx:endx] = imcp[yval, startx:endx]
        else:
            # pdb.set_trace()
            fsum = np.sum(markCoord1[:, 0] - markCoord2[:, 0], dtype=np.int)
            if fsum >= 0:
                markCoord1, markCoord2 = markCoord2, markCoord1
            #
            for i in range(markCoord1.shape[0]):
                xval = markCoord1[i, 1]
                starty = markCoord1[i, 0]
                endy = markCoord2[i, 0]
                mask[starty:endy, xval] = imcp[starty:endy, xval]
        #
        # pdb.set_trace()
        imslice = self.crop_zeros(mask)
        return imslice

    def sliceImageWithMarkCoordPair(self, image: np.ndarray,
                                    markCoord1: np.ndarray,
                                    markCoord2: np.ndarray,
                                    colSlice: bool,
                                    isUpTo: bool) -> np.ndarray:
        "Slice image with mark coordinate pair"
        imcp = np.copy(image)
        # pdb.set_trace()
        assert markCoord1.shape[1] == 2  # [y,x], [y2,x2], etc
        assert markCoord2.shape[1] == 2
        if markCoord1.shape[0] != markCoord2.shape[0]:
            # pdb.set_trace()
            markCoord1, markCoord2 = self.matchMarkCoordPairLength(markCoord1,
                                                                   markCoord2,
                                                                   colSlice,
                                                                   isUpTo)
        #
        imcp = self.swapAndSliceMarkCoordPair(markCoord1, markCoord2,
                                              imcp, colSlice)
        return imcp

    def crop_zeros(self, img: np.ndarray([], dtype=np.uint8)):
        "Crop out zeros from image sides"
        img_cp = img.copy()
        #
        image_col = img_cp.shape[1]
        image_row = img_cp.shape[0]
        #
        delete_list = []
        for col in range(image_col):
            if np.sum(img_cp[:, col], dtype="uint32") == 0:
                delete_list.append(col)
            #
        #
        img_cp = np.delete(arr=img_cp,
                           obj=delete_list,
                           axis=1)
        #
        delete_list = []
        #
        for row in range(image_row):
            if np.sum(img_cp[row, :], dtype="int32") == 0:
                delete_list.append(row)
            #
        img_cp = np.delete(arr=img_cp,
                           obj=delete_list,
                           axis=0)
        #
        return img_cp

    def getMarkCoordinates4Point(self, img: np.ndarray([], dtype=np.uint8),
                                 point1: (int, int),
                                 isUpTo: bool,
                                 colSlice: bool,
                                 thresh: int, mark_color: (int, int, int)):
        "Obtain mark coordinates from image"
        markedImage, mask, sliceImage, beforeAfter = self._markSeam4Point(
            img=img, point1=point1, isUpTo=isUpTo, colSlice=colSlice,
            thresh=thresh, mark_color=mark_color)

        maskImage = np.zeros_like(img, dtype=np.bool)
        maskImage1 = self.addPointSlice2Image(img=maskImage, point=point1,
                                              beforeAfterCoord=beforeAfter,
                                              imgSlice=mask,
                                              colSlice=colSlice,
                                              isUpTo=isUpTo)

        # obtaining mark coordinates from image mask
        m1index = self.getMarkCoordinates(maskImage1)
        if colSlice is False:
            m1index = np.rot90(m1index, 3, (0, 1))
        return m1index

    def getMarkImageWithCoordinates(self, img, point1: (int, int),
                                    isUpTo: bool,
                                    colSlice: bool,
                                    thresh: int,
                                    mark_color: (int, int, int)):
        "Obtain mark coordinates and marked image"
        markedImage, mask, sliceImage, beforeAfter = self._markSeam4Point(
            img=img, point1=point1, isUpTo=isUpTo, colSlice=colSlice,
            thresh=thresh, mark_color=mark_color)
        maskImage = np.zeros_like(img, dtype=np.bool)
        maskImage1 = self.addPointSlice2Image(img=maskImage, point=point1,
                                              beforeAfterCoord=beforeAfter,
                                              imgSlice=mask,
                                              colSlice=colSlice,
                                              isUpTo=isUpTo)
        markedFullImage = self.addPointSlice2Image(
            img=img.copy(),
            point=point1,
            beforeAfterCoord=beforeAfter,
            imgSlice=markedImage,
            colSlice=colSlice,
            isUpTo=isUpTo)

        # obtaining mark coordinates from image mask
        m1index = self.getMarkCoordinates(maskImage1)
        if colSlice is False:
            m1index = np.rot90(m1index, 3, (0, 1))
        return m1index, markedFullImage

    def _markSeam4Point(self, img: np.ndarray([], dtype=np.uint8),
                        point1: (int, int),
                        isUpTo: bool,
                        colSlice: bool,
                        thresh: int,
                        mark_color: (int, int, int)) -> np.ndarray:
        """
        Mark the seam for a given point

        Description
        ------------
        Simple strategy. We slice the image from the given point using
        a threshold value for the sliced area.
        Then mark the seam on that area.
        """
        imcp = img.copy()
        slice1 = self.sliceOnPoint(imcp, point1,
                                   thresh=thresh,
                                   colSlice=colSlice,
                                   isUpTo=isUpTo)
        sl1 = slice1[0]  # image slice
        ba1 = slice1[1]  # before, after coord
        if colSlice is True:
            m1, mask1 = self.mark_column(sl1, mark_color=mark_color)
        else:
            m1, mask1 = self.mark_row(sl1, mark_color=mark_color)
        # m1 == marked image
        return m1, mask1, sl1, ba1
        # adding marked masks back to the image mask

    def markSeam4Point(self, img: np.ndarray([], dtype=np.uint8),
                       point1: (int, int),
                       isUpTo: bool,
                       thresh: int,
                       colSlice: bool,
                       mark_color: (int, int, int)):
        "Mark seam for point"
        markedSlice, mask, sliceImage, beforeAfter = self._markSeam4Point(
            img=img, point1=point1, isUpTo=isUpTo, colSlice=colSlice,
            thresh=thresh, mark_color=mark_color)
        markedImage = self.addPointSlice2Image(img=img, point=point1,
                                               beforeAfterCoord=beforeAfter,
                                               imgSlice=markedSlice,
                                               colSlice=colSlice,
                                               isUpTo=isUpTo)
        return markedImage

    def makePairsFromPoints(self, plist: list,
                            colSlice: bool,
                            isXFirst=False):
        "Make pairs from points by ordering them according to x or y"
        if colSlice is True:
            if isXFirst is False:
                plist.sort(key=lambda p: p[1])
            else:
                plist.sort(key=lambda p: p[0])
        else:
            if isXFirst is False:
                plist.sort(key=lambda p: p[0])
            else:
                plist.sort(key=lambda p: p[1])
        #
        pairs = []
        for i in range(len(plist)):
            if i+1 < len(plist):
                p1 = plist[i]
                p2 = plist[i+1]
                pairs.append((p1, p2))
        return pairs

    def prepDirection(self, direction: str):
        "Prepare direction"
        colSlice = True
        isUpTo = False
        if direction == "down":
            colSlice = True
            isUpTo = False
        elif direction == "up":
            colSlice = True
            isUpTo = True
        elif direction == "right":
            colSlice = False
            isUpTo = False
        elif direction == "left":
            colSlice = False
            isUpTo = True

        return colSlice, isUpTo

    def prepImageWithParams(self, img, plist,
                            direction):
        "Prepare image and point list with respect to the direction"
        imcp = img.copy()
        if direction == 'right':
            # rotate points and image
            imcp = np.rot90(imcp, 1, (0, 1))
            plist = np.rot90(plist, 1, (0, 1))
            plist = [tuple(k) for k in plist.T.tolist()]
        elif direction == 'left':
            imcp = np.rot90(imcp, 1, (0, 1))
            plist = np.rot90(plist, 1, (0, 1))
            plist = [tuple(k) for k in plist.T.tolist()]

        colSlice, isUpTo = self.prepDirection(direction)
        return imcp, plist, isUpTo, colSlice

    def markPointSeam(self, img, point, direction="down",
                      mark_color=[0, 255, 0],
                      thresh=2):
        "Mark seam passes around the point region"
        colSlice, isUpTo = self.prepDirection(direction)
        markedImage = self.markSeam4Point(img.copy(), point, isUpTo, thresh,
                                          colSlice, mark_color)
        return markedImage

    def markPointSeamWithCoordinate(self, img, point, direction='down',
                                    mark_color=[0, 255, 0],
                                    thresh=2):
        "Get mark and coordinate"
        colSlice, isUpTo = self.prepDirection(direction)
        coord, markedImage = self.getMarkImageWithCoordinates(
            img, point, isUpTo, colSlice, thresh, mark_color)
        return markedImage, coord

    def markPointListSeam(self, img, plist: dict, mark_color=[0, 255, 0]):
        """
        Mark seam that passes through the regions of each point

        Description
        -------------

        We assume that each key value pair of the plist, contains
        the following pairs in their values:
        'threshold': int,
        'coordinates': (int, int),
        'x': int,
        'y': int,
        'direction': str
        'color': [int, int, int]
        """
        imcp = img.copy()
        for i, point in plist.items():
            direction = point['direction']
            thresh = point['threshold']
            point_coord = (point['y'], point['x'])
            imcp = self.markPointSeam(imcp,
                                      point_coord,
                                      direction=direction,
                                      thresh=thresh,
                                      mark_color=mark_color)
        #
        return imcp

    def getPointSeamCoordinate(self, img, point,
                               direction="down",
                               thresh=2, mark_color=[0, 255, 0]):
        "Get coordinates of the mark that passes around point region"
        colSlice, isUpTo = self.prepDirection(direction)
        coords = self.getMarkCoordinates4Point(img.copy(), point, isUpTo,
                                               colSlice, thresh, mark_color)
        return coords

    def getPointListSeamCoordinate(self, img, plist,
                                   direction="down",
                                   thresh=2, mark_color=[0, 255, 0]):
        "Get mark coordinates associated to each point region"
        imcp, plist, isUpTo, colSlice = self.prepImageWithParams(img,
                                                                 plist,
                                                                 direction)
        coords = []
        for point in plist:
            coord = self.getMarkCoordinates4Point(img, point, isUpTo,
                                                  colSlice, thresh, mark_color)
            coords.append({"point": point,
                           "markCoordinates": coord})

        return coords

    def makeCoordGroups(self, pointDataCoords: dict):
        "make coordinate groups based on carve directions"
        groups = {"up":[], "down": [], "left": [], "right": []}
        for i, pointData in pointDataCoords.items():
            groups[pointData['direction']].append(pointData)
        return groups

    def segmentImageWithPointListSeamCoordinate(self,
                                                coords: dict,
                                                image):
        "Segment the image using mark coordinates of a point list"
        groups = self.makeCoordGroups(coords)
        segment_groups = {"up": [], "down": [], "left": [], "right": []}
        for groupDirection, pointDataCoords in groups.items():
            # pdb.set_trace()
            colSlice, isUpTo = self.prepDirection(groupDirection)
            plist = [(pointData['y'],
                      pointData['x']) for pointData in pointDataCoords]
            pointCoordMap = {
                (pointData['y'],
                 pointData['x']
                 ):pointData['seamCoordinates'] for pointData in pointDataCoords
            }
            pairs = self.makePairsFromPoints(plist, colSlice,
                                             isXFirst=False)
            segments = []
            for pair in pairs:
                point1 = pair[0]
                point2 = pair[1]
                coord1 = pointCoordMap[point1]
                coord2 = pointCoordMap[point2]
                segment = self.sliceImageWithMarkCoordPair(image, coord1, 
                                                           coord2, colSlice,
                                                           isUpTo)
                segments.append(segment)
            #
            segment_groups[groupDirection] = segments
        #
        return segment_groups

    def segmentPageWithPoints(self, img: np.ndarray([], dtype=np.uint8),
                              plist: [],
                              direction='down',  # allowed values
                              # down/up/left/right
                              mark_color=[0, 255, 0],
                              thresh=2):
        assert thresh >= 0 and thresh <= 100

        if (direction != 'down' and
            direction != 'up' and
            direction != 'left' and
                direction != 'right'):
            raise ValueError(
                'unknown direction, list of known directions {0}'.format(
                    str(['down', 'up', 'left', 'right'])
                )
            )
        #
        imcp, plist, isUpTo, colSlice = self.prepImageWithParams(img,
                                                                 plist,
                                                                 direction)
        # let's make the point pairs
        pairs = self.makePairsFromPoints(plist, colSlice)
        segments = [
            self.getSegmentFromPoints(imcp,
                                      isUpTo=isUpTo,
                                      thresh=thresh,
                                      mark_color=mark_color,
                                      colSlice=colSlice,
                                      point1=pair[0],
                                      point2=pair[1])
            for pair in pairs
        ]
        return segments

    def segmentWithPoints(self):
        return self.segmentPageWithPoints(img=self.img,
                                          plist=self.plist,
                                          thresh=self.thresh,
                                          mark_color=self.mark_color,
                                          direction=self.direction)
