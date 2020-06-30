# mark seams in texts
# author: Kaan Eraslan
# purpose: Project circular or elliptical image coordinates along
# a line, so that seam marking can work on it.

import numpy as np
from PIL import Image, ImageDraw


def sliceImageWithMask(image: np.ndarray, mask: np.ndarray):
    "Slice image using a boolean mask"
    return np.where(mask, image, 255)


def sliceCoordinatesFromMask(mask: np.ndarray):
    "Get coordinates from mask"
    return np.argwhere(mask)


def sliceShapeFromImage(image: np.ndarray,
                        fn: lambda x: x,
                        kwargs,
                        withCoordinate=False):
    mask = np.zeros_like(image, dtype=np.uint8)
    img = Image.fromarray(mask)
    fn(img, **kwargs)
    imgarr = np.array(img)
    mask_bool = imgarr == 255
    if withCoordinate is False:
        return sliceImageWithMask(image, mask_bool)
    else:
        return (sliceImageWithMask(image, mask_bool),
                sliceCoordinatesFromMask(mask_bool))


def findCenterInBbox(bbox: {"x1": int, "y1": int,
                            "x2": int, "y2": int}) -> {"x": int, "y": int}:
    "Find center point in bbox"
    return {"x": (bbox['x1'] + bbox['x2']) // 2,
            "y": (bbox['y1'] + bbox['y2']) // 2}


def drawEllipseWithBbox(image: np.ndarray,
                        bbox: {"x1": int, "y1": int, "x2": int, "y2": int}):
    "Slice an ellipse from an image"
    imdraw = ImageDraw.Draw(image)
    imdraw.ellipse([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                   fill="white")


def sliceEllipseFromImage(image: np.ndarray,
                          bbox: {"x1": int, "y1": int,
                                 "x2": int, "y2": int},
                          withCoord=False):
    params = {"bbox": bbox}
    return sliceShapeFromImage(image,
                               drawEllipseWithBbox,
                               params, withCoord)


def cutEllipse2Half(image: np.ndarray,
                    bbox: {"x1": int, "y1": int,
                           "x2": int, "y2": int},
                    withMask=False):
    """
    Cut ellipse in half vertically

    For first half we paint the second half in white.
    For the second half we paint the first half in white
    """
    ellipseImage, ellipseCoords = sliceEllipseFromImage(image,
                                                        bbox,
                                                        withCoord=True)
    rownb, colnb, pixelnb = ellipseImage.shape
    xindx = ellipseCoords[:, 1]
    centerPoint = findCenterInBbox(bbox)
    centerX = centerPoint['x']
    firstHalfEllipseCoords = ellipseCoords[xindx <= centerX]
    secondHalfEllipseCoords = ellipseCoords[xindx > centerX]
    yindx1 = firstHalfEllipseCoords[:, 0]
    yindx2 = secondHalfEllipseCoords[:, 0]
    xindx1 = firstHalfEllipseCoords[:, 1]
    xindx2 = secondHalfEllipseCoords[:, 1]
    firstHalfEllipse = ellipseImage.copy()
    secondHalfEllipse = ellipseImage.copy()
    firstHalfEllipse[yindx2, xindx2, :] = 255
    secondHalfEllipse[yindx1, xindx1, :] = 255
    if not withMask:
        return firstHalfEllipse, secondHalfEllipse
    else:
        immask = np.zeros_like(ellipseImage, dtype=np.uint8)
        firstHalfMask = immask.copy()
        secondHalfMask = immask.copy()
        firstHalfMask[yindx1, xindx1, :] = 255
        secondHalfMask[yindx2, xindx2, :] = 255
        return (firstHalfEllipse, firstHalfMask,
                secondHalfEllipse, secondHalfMask)


def appendSecondHalf2FirstHalfEllipse(firstHalf: np.ndarray,
                                      firstHalfMask: np.ndarray,
                                      secondHalf: np.ndarray,
                                      secondHalfMask: np.ndarray,
                                      centerPoint: {"x": int, "y": int},
                                      bbox: {"x1": int, "y1": int,
                                             "x2": int, "y2": int},
                                      shapePixelValue=255,
                                      withMask=False):
    """
    Append the second half of the ellipse to first half

    Description
    ------------

    We assume that the ellipse is vertically cut.
    We take the half at the right side of the center, push it down, and move
    it to left, using the center and the bbox as a reference.
    Center gives us the horizontal offset for moving the half to left.
    Bbox gives us the vertical offset for pushing it down

    Parameters
    -----------

    firstHalf: image containing first half of the ellipse
    firstHalfMask: image containing first half of the ellipse where ellipse is
    white/black and the background color is inverse of the ellipse
    secondHalf: image containing second half of the ellipse
    secondHalfMask: image containing second half of the ellipse where ellipse
    is white/black and the background color is its inverse.
    centerPoint: represents the center of the ellipse
    bbox: bounding box of the entire ellipse
    shapePixelValue: pixel value of the shape inside the masks
    """
    rownb, colnb, pixelnb = firstHalf.shape
    verticalOffsetValue = max(bbox['y1'], bbox['y2']) - min(bbox['y1'],
                                                            bbox['y2'])
    # horizontalOffsetValue = centerPoint['x'] - min(bbox['x1'], bbox['x2'])
    horizontalOffsetValue = 0
    secondHalfMask2 = secondHalfMask.copy()
    secondHalf2 = secondHalf.copy()
    secondHalfMask2 = np.flipud(secondHalfMask2)
    secondHalf2 = np.flipud(secondHalf2)
    mask_bool1 = firstHalfMask == shapePixelValue
    mask_bool2 = secondHalfMask2 == shapePixelValue
    firstHalfCoordinates = sliceCoordinatesFromMask(mask_bool1)
    fyindx = firstHalfCoordinates[:, 0]
    fxindx = firstHalfCoordinates[:, 1]
    fzindx = firstHalfCoordinates[:, 2]
    secondHalfCoordinates = sliceCoordinatesFromMask(mask_bool2)
    syindx = secondHalfCoordinates[:, 0]
    sxindx = secondHalfCoordinates[:, 1]
    szindx = secondHalfCoordinates[:, 2]
    syindx2 = syindx + verticalOffsetValue
    sxindx2 = sxindx - horizontalOffsetValue
    synteticImage = np.full((rownb * 2, colnb, pixelnb), 255, dtype=np.uint8)
    firstHalfSynteticImage = synteticImage.copy()
    appendedSynteticImage = synteticImage.copy()
    firstHalfSynteticImage[fyindx, fxindx,
                           fzindx] = firstHalf[fyindx, fxindx, fzindx]
    appendedSynteticImage[fyindx, fxindx,
                          fzindx] = firstHalf[fyindx, fxindx, fzindx]
    appendedSynteticImage[syindx2,
                          sxindx2, szindx] = secondHalf2[syindx, sxindx,
                                                         szindx]
    if withMask:
        synteticImageMask = np.zeros((rownb * 2, colnb, pixelnb), dtype=np.int)
        firstHalfSynteticImageMask = synteticImageMask.copy()
        firstHalfSynteticImageMask[fyindx, fxindx, fzindx] = shapePixelValue
        appendedSynteticImageMask = synteticImageMask.copy()
        firstHalfSynteticImageMask[fyindx, fxindx, fzindx] = 255
        appendedSynteticImageMask[fyindx, fxindx, fzindx] = 255
        appendedSynteticImageMask[syindx2, sxindx2, szindx] = 255
        return (firstHalfSynteticImage, firstHalfSynteticImageMask,
                appendedSynteticImage, appendedSynteticImageMask,
                verticalOffsetValue, horizontalOffsetValue,
                (rownb, colnb, pixelnb))
    else:
        return (firstHalfSynteticImage, appendedSynteticImage,
                verticalOffsetValue, horizontalOffsetValue,
                (rownb, colnb, pixelnb))


def putSecondHalf2OriginalPosition(appendedSynteticImage: np.ndarray,
                                   firstHalfMask: np.ndarray,
                                   appendedMask: np.ndarray,
                                   secondHalfMask: np.ndarray,
                                   imshape: (int, int, int),
                                   shapePixelValue=255,
                                   withMask=False):
    "Put second half to its original position using masks"
    mask_bool1 = firstHalfMask == shapePixelValue
    mask_bool2 = secondHalfMask == shapePixelValue
    firstHalfCoords = sliceCoordinatesFromMask(mask_bool1)
    secondHalfCoords = sliceCoordinatesFromMask(mask_bool2)
    fyindx = firstHalfCoords[:, 0]
    fxindx = firstHalfCoords[:, 1]
    fzindx = firstHalfCoords[:, 2]
    syindx = secondHalfCoords[:, 0]
    minsy, maxsy = syindx.min(), syindx.max()
    sxindx = secondHalfCoords[:, 1]
    minsx, maxsx = sxindx.min(), sxindx.max()
    szindx = secondHalfCoords[:, 2]
    origImg = np.full(imshape, 255, dtype=np.uint8)
    origImg[fyindx, fxindx, fzindx] = appendedSynteticImage[fyindx,
                                                            fxindx,
                                                            fzindx]
    appendedMask2 = appendedMask.copy()
    appendedSynteticImage2 = appendedSynteticImage.copy()
    appendedMask2[fyindx, fxindx, fzindx] = 0
    appendedSynteticImage2[fyindx, fxindx, fzindx] = 255
    appendedMask2F = np.flipud(appendedMask2)
    appendedSynteticImage2F = np.flipud(appendedSynteticImage2)
    mask_bool2 = appendedMask2F == shapePixelValue
    secondHalfCoords = sliceCoordinatesFromMask(mask_bool2)
    syindx_off = secondHalfCoords[:, 0]
    miny, maxy = syindx_off.min(), syindx_off.max()
    sxindx_off = secondHalfCoords[:, 1]
    minx, maxx = sxindx_off.min(), sxindx_off.max()
    szindx_off = secondHalfCoords[:, 2]
    synteticImage = appendedSynteticImage2F[miny:maxy+1,
                                            minx:maxx+1, :]
    synteticMask = appendedMask2F[miny:maxy+1,
                                  minx:maxx+1, :]
    origImg[minsy:maxsy+1, minsx:maxsx+1, :] = synteticImage
    if withMask:
        origImgMask = np.zeros_like(origImg, dtype=np.int)
        origImgMask[fyindx, fxindx, fzindx] = shapePixelValue
        origImgMask[syindx, sxindx, szindx] = shapePixelValue
        return (origImg, origImgMask)
    else:
        return origImg


def cutAndAppendEllipse(image: np.ndarray,
                        bbox: {"x1": int, "y1": int,
                               "x2": int, "y2": int}):
    "Join second half to first half"
    centerX = (bbox['x1'] + bbox['x2']) // 2
    centerY = (bbox['y1'] + bbox['y2']) // 2
    (firstHalf, firstHalfMask,
     secondHalf, secondHalfMask) = cutEllipse2Half(image, bbox, withMask=True)
    rownb, colnb, pixelnb = firstHalf.shape
    mask = np.zeros((rownb * 2, colnb, pixelnb), dtype=np.uint8)


def drawThickLineWithBbox(image: np.ndarray,
                          bbox: {"x1": int, "y1": int, "x2": int, "y2": int},
                          width: int):
    "Draw line with bbox"
    imdraw = ImageDraw.Draw(image)
    imdraw.line([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                fill="white", width=width)


def sliceThickLineFromImage(image: np.ndarray,
                            bbox: {"x1": int, "y1": int,
                                   "x2": int, "y2": int},
                            line_width=3, withCoord=False):
    "Slice thick line from image"
    assert line_width > 1
    params = {"bbox": bbox, "width": line_width}
    return sliceShapeFromImage(image, drawThickLineWithBbox,
                               params, withCoord)


def drawPolygonWithBbox(image: np.ndarray,
                        coords: [{"x": int, "y": int},
                                 {"x": int, "y": int}]):
    "Draw polygon with bbox"
    imdraw = ImageDraw.Draw(image)
    coordlist = [(coord['x'], coord['y']) for coord in coords]
    imdraw.polygon(coordlist, fill="white")


def slicePolygonFromImage(image: np.ndarray,
                          coords: [{"x": int, "y": int},
                                   {"x": int, "y": int}],
                          withCoord=False):
    "Slice polygon defined by bbox"
    params = {"coords": coords}
    return sliceShapeFromImage(image, drawPolygonWithBbox,
                               params, withCoord)
