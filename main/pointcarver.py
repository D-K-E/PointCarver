# mark seams in texts
# author: Kaan Eraslan
# Implementation in part taken from the link:
# https://karthikkaranth.me/blog/implementing-seam-carving-with-python/):

# Author in the link: Karthik Karanth

import numpy as np  # array/matrix manipulation
import scipy.ndimage as nd  # operate easily on image matrices


class SeamMarker:
    def __init__(self,
                 img: np.ndarray([], dtype=np.uint8),
                 plist=[],
                 thresh=10,
                 direction='down'):
        self.img = img
        self.plist = plist
        self.direction = direction
        self.thresh = thresh
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

    def sliceOnPoint(self, img: np.ndarray([], dtype=np.uint8),
                     point: (int, int),
                     isUpTo=False,
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

        # column wise slicing

        col_nb = img.shape[1]
        pointCol = point[1]
        colBefore, colAfter = self.expandPointCoordinate(ubound=col_nb,
                                                         coord=pointCol,
                                                         thresh=thresh)
        before, after = colBefore, colAfter
        if isUpTo is False:
            imgSlice = img[point[0]:, colBefore:colAfter]
        else:
            imgSlice = img[:point[0], colBefore:colAfter]

        return imgSlice, (before, after)

    def addPointSlice2Image(self,
                            img: np.ndarray([], dtype=np.uint8),
                            point: (int, int),  # y, x
                            beforeAfterCoord: (int, int),
                            imgSlice: np.ndarray([], dtype=np.uint8),
                            isUpTo: bool,
    ):
        "Add sliced zone back to image"
        imcp = img.copy()
        # pdb.set_trace()
        before, after = beforeAfterCoord
        if isUpTo is False:
            imcp[point[0]:, before:after] = imgSlice
        else:
            imcp[:point[0], before:after] = imgSlice
        return imcp

    def getMarkCoordinates(self,
                           markedMask: np.ndarray([], dtype=np.bool)):
        "Get marked coordinates from the mask"
        indexArray = markedMask.nonzero()
        indexArray = np.array(indexArray)
        indexArray = indexArray.T
        # indexArray[0] == [rowPosition, colPosition, colorPosition]
        return indexArray

    def placePointsOnMask(self,
                          row1: int,
                          row2: int,
                          col1: int,
                          col2: int,
                          mask: np.ndarray([], dtype=np.uint8),
                          img: np.ndarray([], dtype=np.uint8),
    ):
        "Place points on the mask by comparing row and col position"
        # if column segmentation rows should be
        # equal if not cols should be equal
        if row1 == row2:
            if col1 < col2:
                mask[row1, col1:col2, :] = img[row1, col1:col2, :]
            elif col2 < col1:
                mask[row1, col2:col1, :] = img[row1, col2:col1, :]
            elif col2 == col1:
                mask[row1, col2, :] = img[row1, col2, :]
        return mask

    def sliceImageFromSameSizeMarks(self,
                                    img: np.ndarray([], dtype=np.uint8),
                                    markIndex1: np.ndarray([], dtype=np.int32),
                                    markIndex2: np.ndarray([], dtype=np.int32),
    ):
        "Slice image from same size marks"
        mask = np.zeros_like(img, dtype=np.uint8)
        markrow1 = markIndex1.shape[0]
        for indxr in range(markrow1):
            row1, col1, color1 = markIndex1[indxr]
            row2, col2, color2 = markIndex2[indxr]

            # check if the slice is rowwise
            # if column segmentation rows should be
            # equal if not cols should be equal
            if row1 != row2:
                continue
            mask = self.placePointsOnMask(row1, row2,
                                          col1, col2,
                                          mask, img,
            )
        return mask

    def sliceImageFromDifferentSizeMarks(self, img: np.ndarray([], dtype=np.uint8),
                                         markIndex1: np.ndarray([], dtype=np.int32),
                                         markIndex2: np.ndarray([], dtype=np.int32),
    ):
        "Slice Image with different size mark indices"
        mask = np.zeros_like(img, dtype=np.uint8)
        indexrow1 = markIndex1.shape[0]
        indexrow2 = markIndex2.shape[0]

        # iterate over the rows of index
        for indxr1 in range(indexrow1):
            for indxr2 in range(indexrow2):

                row1, col1, color1 = markIndex1[indxr1]
                row2, col2, color2 = markIndex2[indxr2]

                # if column segmentation rows should be
                # equal
                if row1 != row2:
                    continue
                mask[row1, col1:col2, :] = img[row1, col1:col2, :]

        return mask

    def sliceImageWithMarks(self,
                            img: np.ndarray([], dtype=np.uint8),
                            markIndex1: np.ndarray([], dtype=np.int32),
                            markIndex2: np.ndarray([], dtype=np.int32),
    ):
        "Slice the image using marks"

        # markrow1 = markIndex1.shape[0]
        # markrow2 = markIndex2.shape[0]
        # markrow1 = markIndex1[-1][0]  # should be the last row
        # markrow2 = markIndex2[-1][0]  # should be the last row

        # markcol1 = markIndex1[-1][1]  # should be the last col
        # markcol2 = markIndex2[-1][1]  # should be the last col
        # pdb.set_trace()

        # if colSlice is True:
        #     assert markrow1 == markrow2, "Expected same row shape"\
        #         "but got: {0}, {1}".format(
        #             markIndex1.shape,
        #             markIndex2.shape
        #         )
        # make sure marks come from same image
        # or at least image with same size
        indexrow1 = markIndex1.shape[0]
        indexrow2 = markIndex2.shape[0]

        if indexrow1 == indexrow2:
            mask = self.sliceImageFromSameSizeMarks(img=img,
                                                    markIndex1=markIndex1,
                                                    markIndex2=markIndex2,
            )
        else:
            mask = self.sliceImageFromDifferentSizeMarks(img=img,
                                                         markIndex1=markIndex1,
                                                         markIndex2=markIndex2,
            )
        return mask

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

    def getSegmentFromPoints(self,
                             img: np.ndarray([], dtype=np.uint8),
                             point1: (int, int),
                             point2: (int, int),
                             isUpTo: bool,
                             colSlice: bool,
                             thresh: int,
                             mark_color: (int, int, int)) -> ():
        "Get column from given points and the direction"
        # check whether points are consecutive
        imcp = img.copy()
        p1Col = point1[1]
        p2Col = point2[1]
        assert p2Col >= p1Col

        # Slicing on point
        slice1 = self.sliceOnPoint(imcp, point1,
                                   thresh=thresh,
                                   isUpTo=isUpTo)
        sl1 = slice1[0]
        ba1 = slice1[1]

        # Slicing the second point
        slice2 = self.sliceOnPoint(imcp, point2,
                                   thresh=thresh,
                                   isUpTo=isUpTo)
        sl2 = slice2[0]
        ba2 = slice2[1]

        # getting marks
        m1, mask1 = self.mark_column(sl1, mark_color=mark_color)
        m2, mask2 = self.mark_column(sl2, mark_color=mark_color)

        # adding marked masks back to the image mask
        maskImage = np.zeros_like(imcp, dtype=np.bool)
        maskImage1 = self.addPointSlice2Image(maskImage, point1,
                                              ba1, mask1,
                                              isUpTo=isUpTo)

        # obtaining mark coordinates from image mask
        m1index = self.getMarkCoordinates(maskImage1)

        # same for the second mark
        maskImage2 = self.addPointSlice2Image(maskImage, point2,
                                              ba2, mask2,
                                              isUpTo=isUpTo)
        m2index = self.getMarkCoordinates(maskImage2)
        im1mask = self.sliceImageWithMarks(imcp,
                                           m1index, m2index,
        )
        if colSlice is False:
            plist = [point1, point2]
            im1mask = np.rot90(im1mask, 3, (0, 1))
            plist = np.rot90(plist, 3, (0, 1))
            plist = [tuple(k) for k in plist.T.tolist()]
            point1 = plist[0]
            point2 = plist[1]

        im1mask = self.crop_zeros(im1mask)
        return (im1mask, point1, point2)

    def makePairsFromPoints(self, plist: [],
                            direction: str):
        'Make point pairs after sorting point list with given direction'
        if (direction != 'down' and
            direction != 'up' and
            direction != 'left' and
            direction != 'right'):
            raise ValueError('unknown direction')

        if direction == 'down' or direction == 'up':
            plist.sort(key=lambda p: p[0])
        else:
            plist.sort(key=lambda p: p[1])

        pairs = []
        for i in range(len(plist)):
            if i+1 < len(plist):
                p1 = plist[i]
                p2 = plist[i+1]
                pairs.append((p1, p2))
        return pairs

    def segmentPageWithPoints(self, img: np.ndarray([], dtype=np.uint8),
                              plist: [],
                              direction='down',  # allowed values down/up/left/right
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
        imcp = img.copy()
        if direction == 'down':
            colSlice = True
            isUpTo = False
        elif direction == 'up':
            colSlice = True
            isUpTo = True
        elif direction == 'right':
            # rotate points and image
            colSlice = False
            imcp = np.rot90(imcp, 1, (0, 1))
            plist = np.rot90(plist, 1, (0, 1))
            plist = [tuple(k) for k in plist.T.tolist()]
            isUpTo = False
        elif direction == 'left':
            colSlice = False
            imcp = np.rot90(imcp, 1, (0, 1))
            plist = np.rot90(plist, 1, (0, 1))
            plist = [tuple(k) for k in plist.T.tolist()]
            isUpTo = True
        #
        # let's make the point pairs
        pairs = self.makePairsFromPoints(plist, direction)
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
