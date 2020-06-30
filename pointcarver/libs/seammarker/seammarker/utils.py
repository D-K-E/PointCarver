from PIL import Image
import numpy as np
import json

# From stack overflow


def getConsecutive1D(data: np.ndarray,
                     stepsize=1,
                     only_index=False):
    "Get consecutive values from 1d array"
    assert data.ndim == 1
    indices = np.argwhere((data[1:] - data[:-1]) != stepsize)
    indices = indices.T[0] + 1  # include the range
    if only_index is False:
        subarrays = np.split(data, indices, axis=0)
        return subarrays, indices
    else:
        return indices


def getDiffDirection(stepsize: int,
                     direction: str):
    "create 2d diff vec using stepsize"
    direction = direction.lower()
    assert direction in ["vertical", "horizontal",
                         "diagonal-l", "diagonal-r"]
    if direction == "vertical":
        rowdiff, coldiff = stepsize, 0
    elif direction == "horizontal":
        rowdiff, coldiff = 0, stepsize
    elif direction == "diagonal-l":
        rowdiff, coldiff = stepsize, stepsize
    elif direction == "diagonal-r":
        rowdiff, coldiff = stepsize, -stepsize
    return [rowdiff, coldiff]


def getRowColumnMask(line: np.ndarray, coordarr: np.ndarray) -> np.ndarray:
    """Get row column boolean mask indicating whether line coordinates are in
    coordinate array
    """
    cols = line[:, 1]
    rows = line[:, 0]
    rowbool = np.isin(rows, coordarr[:, 0])
    colbool = np.isin(cols, coordarr[:, 1])
    mask = rowbool & colbool
    return mask


def filterLineCoordsWithCoordinates(line: np.ndarray,
                                    coordarr: np.ndarray) -> np.ndarray:
    "Filter line coordinates using coordinate array"
    mask = getRowColumnMask(line, coordarr)
    line = line[mask, :]
    newline = []
    for i in range(line.shape[0]):
        coord = line[i, :]
        if any(np.equal(coordarr, coord).all(axis=1)):
            newline.append(coord)
    return np.array(newline, dtype=np.int)


def getStraightLinePointsWithSteps(
        point1: {"x": int, "y": int},
        point2: {"x": int, "y": int},
        stepsize=1):
    """
    Get line from points including the points included in the line
    Bresenham's line algorithm adapted from pseudocode in wikipedia:
    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    image should be grayscale

    """
    #  define local variables for readability
    P1X = point1['x']
    P1Y = point1["y"]
    P2X = point2["x"]
    P2Y = point2["y"]

    #  difference and absolute difference between points
    #  used to calculate slope and relative location between points
    diffX = P2X - P1X
    diffXa = np.absolute(diffX, dtype="int32")
    diffY = P2Y - P1Y
    diffYa = np.absolute(diffY, dtype="int32")
    #
    steepx = stepsize
    if P1X < P2X:
        steepx = stepsize
    else:
        steepx = -stepsize
    #
    if P1Y < P2Y:
        steepy = stepsize
    else:
        steepy = -stepsize
    #
    div_term = diffXa
    #
    if diffXa > diffYa:
        div_term = diffXa
    else:
        div_term = -diffYa
        #
    error = div_term / 2
    #
    error2 = 0
    #
    arrival_condition = bool((P1X, P1Y) == (P2X, P2Y))
    #
    line_points = []
    line_points.append([P1Y, P1X])
    #
    while arrival_condition is False:
        error2 = error
        if error2 > -diffXa:
            error = error - diffYa
            P1X = P1X + steepx
            #
        if error2 < diffYa:
            error = error + diffXa
            P1Y = P1Y + steepy
            #
            # Check
        line_points.append([P1Y, P1X])
        arrival_condition = bool((P1Y, P1X) == (P2Y, P2X))
    #
    line_points = np.array(line_points, dtype=np.int)
    return line_points


def getStraightLineWithStepsInZone(point1: {"x": int, "y": int},
                                   point2: {"x": int, "y": int},
                                   zone: np.ndarray,
                                   stepsize=1
                                   ):
    """
    Get line from points including the points included in the line
    Bresenham's line algorithm adapted from pseudocode in wikipedia:
    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    image should be grayscale

    """
    #  define local variables for readability
    P1X = point1['x']
    P1Y = point1["y"]
    P2X = point2["x"]
    P2Y = point2["y"]

    #  difference and absolute difference between points
    #  used to calculate slope and relative location between points
    diffX = P2X - P1X
    diffXa = np.absolute(diffX, dtype="int32")
    diffY = P2Y - P1Y
    diffYa = np.absolute(diffY, dtype="int32")
    #
    steepx = stepsize
    if P1X < P2X:
        steepx = stepsize
    else:
        steepx = -stepsize
    #
    if P1Y < P2Y:
        steepy = stepsize
    else:
        steepy = -stepsize
    #
    div_term = diffXa
    #
    if diffXa > diffYa:
        div_term = diffXa
    else:
        div_term = -diffYa
        #
    error = div_term / 2
    #
    error2 = 0
    #
    arrival_condition = bool((P1X, P1Y) == (P2X, P2Y))
    #
    line_points = []
    line_points.append([P1Y, P1X])
    #
    while arrival_condition is False:
        error2 = error
        if error2 > -diffXa:
            error = error - diffYa
            P1X = P1X + steepx
            #
        if error2 < diffYa:
            error = error + diffXa
            P1Y = P1Y + steepy
            #
            # Check
        line_points.append([P1Y, P1X])
        arrival_condition = bool((P1Y, P1X) == (P2Y, P2X))
    #
    line_points = np.array(line_points, dtype=np.int)
    line_points = filterLineCoordsWithCoordinates(line=line_points,
                                                  coordarr=zone)
    return line_points


def getStraightLineWithoutEnergy(point1: {"x": int, "y": int},
                                 point2: {"x": int, "y": int},
                                 image: np.ndarray,
                                 stepsize=1,
                                 threshold=0):
    """
    Get line from points including the points included in the line
    Bresenham's line algorithm adapted from pseudocode in wikipedia:
    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    image should be grayscale

    Line should have less energy than the threshold

    """
    #  define local variables for readability
    P1X = point1['x']
    P1Y = point1["y"]
    P2X = point2["x"]
    P2Y = point2["y"]

    #  difference and absolute difference between points
    #  used to calculate slope and relative location between points
    diffX = P2X - P1X
    diffXa = np.absolute(diffX, dtype="int32")
    diffY = P2Y - P1Y
    diffYa = np.absolute(diffY, dtype="int32")
    #
    steepx = stepsize
    if P1X < P2X:
        steepx = stepsize
    else:
        steepx = -stepsize
    #
    if P1Y < P2Y:
        steepy = stepsize
    else:
        steepy = -stepsize
    #
    div_term = diffXa
    #
    if diffXa > diffYa:
        div_term = diffXa
    else:
        div_term = -diffYa
        #
    error = div_term / 2
    #
    error2 = 0
    #
    arrival_condition = bool((P1X, P1Y) == (P2X, P2Y))
    #
    line_points = []
    line_points.append([P1Y, P1X])
    #
    while arrival_condition is False:
        error2 = error
        if error2 > -diffXa:
            error = error - diffYa
            P1X = P1X + steepx
            #
        if error2 < diffYa:
            error = error + diffXa
            P1Y = P1Y + steepy
            #
            # Check
        line_points.append([P1Y, P1X])
        imgEnergy = image[P1Y, P1X]
        arrival_condition = bool(
            (P1Y, P1X) == (P2Y, P2X) and imgEnergy <= threshold
        )
    #
    line_points = np.array(line_points, dtype=np.int)
    return line_points


def filterLines4UniqueCoordinates(lines: [np.ndarray]):
    "Add line to lines if line coordinates are not in any of the lines"
    newlines = []
    while lines:
        inline = lines.pop()
        checks = []
        for line in lines:
            mask = getRowColumnMask(inline, coordarr=line)
            check = mask.all()
            checks.append(check)
        checks = np.array(checks, dtype=np.bool)
        if checks.all() is False:
            print("coordinate check false")
            newlines.append(inline)
    #
    return newlines


def getLinesFromCoordinates(coordarr: np.ndarray):
    """Obtain lines from coordinate array"""
    assert len(coordarr.shape) == 2
    assert coordarr.shape[1] == 2
    minRow = coordarr[:, 0].min()
    maxRow = coordarr[:, 0].max()
    minRowCoords = coordarr[coordarr[:, 0] == minRow]
    maxRowCoords = coordarr[coordarr[:, 0] == maxRow]
    minMaxLines = []
    for i in minRowCoords[:, 1]:
        for c in maxRowCoords[:, 1]:
            point1 = {"x": i,
                      "y": minRow}
            point2 = {"x": c,
                      "y": maxRow}
            line = getStraightLineWithStepsInZone(point1,
                                                  point2,
                                                  zone=coordarr)
            # line = filterLineCoordsWithCoordinates(line, coordarr)
            minMaxLines.append(line)
    #
    # minMaxLines = filterLines4UniqueCoordinates(minMaxLines)
    return minMaxLines


def getConsecutive2D(data: np.ndarray,
                     direction: str,
                     stepsize=1,
                     only_index=False):
    "Get consecutive values in horizontal vertical and diagonal directions"
    assert len(data.shape) == 2
    assert data.shape[1] == 2
    diffval = getDiffDirection(stepsize, direction)
    diffarr = data[1:] - data[:-1]
    indices = np.argwhere(diffarr != diffval)
    indices = indices.T
    indices = indices[0] + 1
    if only_index:
        return indices
    else:
        splitdata = np.split(data, indices, axis=0)
        splitdata = [
            data for data in splitdata if data.size > 0 and data.shape[0] > 1
        ]
        return splitdata, indices


def getConsecutive2DSplit(data: np.ndarray,
                          direction: str,
                          stepsize=1,
                          only_index=False):
    ""
    pass

# End stack overflow


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


def assertCond(var, cond: bool, printType=True):
    "Assert condition print message"
    if printType:
        assert cond, 'variable value: {0}\nits type: {1}'.format(var,
                                                                 type(var))
    else:
        assert cond, 'variable value: {0}'.format(var)


def normalizeImageVals(img: np.ndarray):
    ""
    r, c = img.shape[:2]
    flatim = img.reshape((-1))
    #
    normImg = np.interp(flatim,
                        (flatim.min(), flatim.max()),
                        (0, 255),
                        )
    normImg = normImg.astype(np.uint8)
    normImg = normImg.reshape((r, c))
    return normImg

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
