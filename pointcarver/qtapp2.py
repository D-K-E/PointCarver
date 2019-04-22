# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE


from src.seammarker import SeamMarker
from src.utils import readImage, readPoints, parsePoints, stripExt
from src.utils import qt_image_to_array, saveJson
from ui.designerOutput2 import Ui_MainWindow as UIMainWindow

from PIL import Image, ImageQt
from PySide2 import QtGui, QtCore, QtWidgets
import sys
import os
import json
import numpy as np
import io
import base64
import pdb


# some utility function related to interface

def img2str(img: Image) -> str:
    f = io.BytesIO()
    imbin = img.save(f, format='PNG')
    imbin = f.getvalue()
    return str(imbin, 'latin1', 'strict')


def str2img(imdata: str, mode: str, size: (int, int)):
    imbyte = bytes(imdata, 'latin1', 'strict')
    return Image.frombytes(mode=mode, size=size, data=imbyte)


class AppWindowInit(UIMainWindow):
    """
    Initializes the image window
    made in qt designer
    """

    def __init__(self):
        self.main_window = QtWidgets.QMainWindow()
        super().setupUi(self.main_window)
        pass


class PointEditor(QtWidgets.QWidget):
    "Handler for integrating editor with individual point widgets"

    def __init__(self,
                 browserWidget: QtWidgets.QTextBrowser,
                 directionWidget: QtWidgets.QComboBox,
                 thresholdWidget: QtWidgets.QSpinBox,
                 coordsWidget: QtWidgets.QComboBox,
                 colorWidget: QtWidgets.QComboBox,
                 xValueWidget: QtWidgets.QSpinBox,
                 yValueWidget: QtWidgets.QSpinBox,
                 sizeWidget: QtWidgets.QSpinBox,
                 colors={"red": QtCore.Qt.red,
                         "black": QtCore.Qt.black,
                         "green": QtCore.Qt.green,
                         "yellow": QtCore.Qt.yellow,
                         "cyan": QtCore.Qt.cyan,
                         "blue": QtCore.Qt.blue,
                         "gray": QtCore.Qt.gray,
                         "magenta": QtCore.Qt.magenta,
                         "white": QtCore.Qt.white},
                 parent=None):
        self.browserWidget = browserWidget
        self.directionWidget = directionWidget
        self.directionWidget.setEditable(False)
        self.directionWidget.setDuplicatesEnabled(False)

        self.thresholdWidget = thresholdWidget
        self.colorWidget = colorWidget
        self.sizeWidget = sizeWidget
        self.xValSpin = xValueWidget
        self.yValSpin = xValueWidget
        self.coordsCombo = coordsWidget
        self.coordsCombo.setEditable(False)
        self.coordsCombo.setDuplicatesEnabled(False)
        pointDataTemplate = {(None, None): {"color": "",
                                      "direction": "",
                                      "threshold": 1,
                                      "size": 1,
                                      "x": 0,
                                      "y": 0,
                                      }
                             }
        self.clearPointsData = pointDataTemplate
        self.pointsData = {}

        self.colors = colors
        super().__init__(parent=parent)

        # Events
        self.xValSpin.valueChanged.connect(self.setXval2Coords)
        self.yValSpin.valueChanged.connect(self.setYval2Coords)
        self.thresholdWidget.valueChanged.connect(self.setThreshold2Point)
        self.colorWidget.currentTextChanged.connect(
            self.setColor2Point)
        self.directionWidget.currentTextChanged.connect(
            self.setDirection2Point)
        self.sizeWidget.valueChanged.connect(self.setSize2Point)
        self.coordsCombo.currentTextChanged.connect(
            self.showPointCoord)

    def getCurrentCoords(self):
        "Get current coords"
        # x, y
        coords = self.coordsCombo.currentText()
        coords = tuple(coords)
        return coords

    def getPointDataFromWidgets(self, withCoords: bool):
        "set points and related data to widgets"
        color = self.colorWidget.currentText()
        xval = self.xValSpin.value()
        yval = self.yValSpin.value()
        pointSize = self.sizeWidget.value()
        direction = self.directionWidget.currentText()
        threshold = self.thresholdWidget.value()
        if withCoords:
            coords = self.getCurrentCoords()
            return {coords: {"color": color, "pointSize": pointSize,
                             "direction": direction, "threshold": threshold,
                             "x": xval, "y": yval}
                    }
        else:
            return {"color": color, "pointSize": pointSize,
                    "direction": direction, "threshold": threshold}

    def getCurrentPointDataFromWidgets(self):
        "get point data without coords"
        return self.getPointDataFromWidgets(withCoords=False)

    def setVal2Coords(self, val: int, index: int):
        "Sets coordinate value to current coordinates"
        coords = self.getCurrentCoords()
        data = self.getPointDataByPoint(coords)
        if index == 0:
            data['x'] = val
        else:
            data['y'] = val
        self.pointsData.pop(coords, None)
        coords = list(coords)
        coords[index] = val
        coords = tuple(coords)
        self.pointsData[coords] = data
        coordstr = str(tuple(coords))
        self.coordsCombo.setCurrentText(coordstr)

    def setXval2Coords(self):
        "Set x coordinate value to current coordinates"
        val = self.xValSpin.value()
        self.setVal2Coords(val, 0)

    def setYval2Coords(self):
        "Set x coordinate value to current coordinates"
        val = self.yValSpin.value()
        self.setVal2Coords(val, 1)

    def setThreshold2Point(self):
        coords = self.getCurrentCoords()
        data = self.getPointDataByPoint(coords)
        newval = self.thresholdWidget.value()
        data['threshold'] = newval
        self.pointsData[coords] = data

    def setDirection2Point(self):
        coords = self.getCurrentCoords()
        data = self.getPointDataByPoint(coords)
        newval = self.directionWidget.currentText()
        data['direction'] = newval
        self.pointsData[coords] = data

    def setColor2Point(self):
        coords = self.getCurrentCoords()
        data = self.getPointDataByPoint(coords)
        newval = self.colorWidget.currentText()
        data['color'] = newval
        self.pointsData[coords] = data

    def setSize2Point(self):
        coords = self.getCurrentCoords()
        data = self.getPointDataByPoint(coords)
        newval = self.sizeWidget.value()
        data['size'] = newval
        self.pointsData[coords] = data

    def showPointCoord(self):
        coords = self.getCurrentCoords()
        self.showPointData(coords)

    def getPointDataByPoint(self,
                            pointCoords: (int, int)):
        "Get point data from points data by using coords"
        return self.pointsData[pointCoords]

    def showPointData(self, pointCoords: (int, int)):
        "Show point data on widgets"
        data = self.getPointDataByPoint(pointCoords)
        color = data['color']
        direction = data['direction']
        threshold = data['threshold']
        size = data['size']
        x = data['x']
        y = data['y']
        self.directionWidget.setCurrentText(direction)
        self.colorWidget.setCurrentText(color)
        self.thresholdWidget.setValue(threshold)
        self.xValSpin.setValue(x)
        self.yValSpin.setValue(y)
        self.sizeWidget.setValue(size)

    def addPointCoordsWithCurrentData(self,
                                      pointCoords: (int, int)):
        "Set point data from points data by using coords"
        data = self.getCurrentPointDataFromWidgets()
        data['x'] = pointCoords[0]
        data['y'] = pointCoords[1]
        self.pointsData[pointCoords] = data
        self.renderPointsData()

    def setDataFromPointsData(self, pointsData: dict):
        "Set widgets' data from points data"
        self.pointsData = {}
        self.coordsCombo.clear()
        self.directionWidget.clear()
        for coords, data in pointsData.items():
            coordstr = str(coords)
            self.coordsCombo.addItem(coordstr)
            self.directionWidget.addItem(data['direction'])
        #
        self.pointsData = pointsData
        self.renderPointsData()

    def clearData(self):
        self.pointsData = {}
        self.browserWidget.clear()
        self.setDataFromPointsData(self.clearPointsData)

    def renderPointsData(self):
        "render points data in text browser"
        self.browserWidget.clear()
        pointsDataStr = json.dumps(self.pointsData, ensure_ascii=False)
        self.browserWidget.setText(pointsDataStr)


class SceneCanvas(QtWidgets.QGraphicsScene):
    "Mouse events overriding for graphics scene with editor integration"

    def __init__(self,
                 pointEditor: PointEditor,
                 image: QtGui.QPixmap,
                 parent=None):
        self.pointEditor = pointEditor
        self.image = image

    def addPoint2PointEditor(self, point):
        "Add point coords to point editor"
        self.pointEditor.addPointCoordsWithCurrentData(point)

    def drawPointsOnImage(self):
        self.clear()
        image = self.image.copy()
        imw, imh = image.width(), image.height()
        result = QtGui.QPixmap(w=imw, h=imh)
        result.fill(QtCore.Qt.white)
        points = self.pointEditor.pointsData.keys()
        painter = QtGui.QPainter()
        painter.begin(image)
        painter.drawPixmap(0, 0, image)
        for coord, data in self.pointEditor.pointsData.items():
            color = self.pointEditor.colors[data['color']]
            size = self.pointEditor.pointsData[data['size']]
            brush = QtGui.QBrush(color)
            painter.setBrush(brush)
            painter.setPen(color)
            px = coord[0]
            py = coord[1]
            painter.drawEllipse(px, py, size, size)
        #
        painter.end()
        pixmapItem = QtWidgets.QGraphicsPixmapItem(image)
        self.addItem(pixmapItem)

    def drawPixmapImage(self):
        image = self.image.copy()
        self.clear()
        pixmapItem = QtWidgets.QGraphicsPixmapItem(image)
        self.addItem(pixmapItem)

    def mouseDoubleClickEvent(self, event):
        "Overriding double click event"
        point = event.scenePos()
        x, y = point.x(), point.y()
        pointInv = [int(y), int(x)]
        self.addPoint2PointEditor(pointInv)
        self.drawPointsOnImage()


class AppWindowFinal(AppWindowInit):
    "Application window"

    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmapImage = QtGui.QPixmap()
        self.coords = None
        self.coords_direction = None
        self.coords_colSlice = None
        self.coords_thresh = None

        #
        self.assetsdir = ""
        self.imagePoint = {}

        self.colors = {"red": QtCore.Qt.red,
                       "black": QtCore.Qt.black,
                       "green": QtCore.Qt.green,
                       "yellow": QtCore.Qt.yellow,
                       "cyan": QtCore.Qt.cyan,
                       "blue": QtCore.Qt.blue,
                       "gray": QtCore.Qt.gray,
                       "magenta": QtCore.Qt.magenta,
                       "white": QtCore.Qt.white}
        combovals = list(self.colors.keys())
        self.markColorComboBox.addItems(combovals)
        self.directions = ['down', 'up', 'left', 'right']
        self.pointDirectionComboBox.addItems(self.directions)
        self.carveDirComboBox.addItems(self.directions)
        self.pointEditor = PointEditor(
            browserWidget=self.pointsBrowser,
            directionWidget=self.pointDirectionComboBox,
            thresholdWidget=self.thresholdSpin,
            coordsWidget=self.pointCoordsComboBox,
            colorWidget=self.pointColorComboBox,
            xValueWidget=self.pointXSpinBox,
            yValueWidget=self.pointYSpinBox,
            sizeWidget=self.pointSizeSpin,
        )
        self.scene = SceneCanvas(self.pointEditor,
                                 self.image)

        # table widget related
        self.tableWidget.setHorizontalHeaderLabels(["image files", 
                                                    "point files"])

        # hide show widgets
        self.globalThreshCBox.setCheckState(QtCore.Qt.Checked)
        self.globalCarveDirCBox.setCheckState(QtCore.Qt.Checked)

        # Main Window Events
        self.main_window.setWindowTitle("Seam Marker using Points")
        self.main_window.closeEvent = self.closeApp

        self.globalCarveDirCBox.stateChanged.connect(
            self.showGlobalCarveDir)
        self.globalThreshCBox.stateChanged.connect(
            self.showGlobalThresh)

        # Buttons
        self.importBtn.clicked.connect(self.importImagePoints)
        self.loadBtn.clicked.connect(self.loadImage)
        # self.drawPointsBtn.clicked.connect(self.drawEditorPoints)
        # self.carveBtn.clicked.connect(self.markSeamsOnImage)
        # self.savePointsBtn.clicked.connect(self.savePoints)
        # self.saveCoordinatesBtn.clicked.connect(self.saveSeamCoordinates)
        # self.saveBtn.clicked.connect(self.saveSegments)
        # self.saveAllBtn.clicked.connect(self.saveAll)
        self.addPoint2ImageBtn.clicked.connect(self.importPoint)
        self.openPointBtn.clicked.connect(self.setPoints2PointEditor)


    def getPointPathFromImagePath(self, imagePath):
        "Get point file from image path"
        pardir = os.path.dirname(imagePath)  # assetsPath/images
        self.assetsdir = os.path.dirname(pardir)  # assetsPath
        pointsdir = os.path.join(self.assetsdir, 'points')  # assetsPath/points
        imname = os.path.basename(imagePath)
        imNoExt = stripExt(imname)[0]
        pointsName = imNoExt + "-points.json"
        pointsPath = os.path.join(pointsdir, pointsName)
        return pointsPath

    def checkPointPath(self, pointPath):
        "Check point path"
        return os.path.isfile(pointPath)

    def findPointPathFromImagePath(self, imagePath):
        "Find point path from image path if it exists"
        pointPath = self.getPointPathFromImagePath(imagePath)
        res = None
        if self.checkPointPath(pointPath):
            res = pointPath
        return res

    def addImage2Table(self, imagePath):
        "Add image path to table"
        im = {}
        im['path'] = imagePath
        im['name'] = os.path.basename(imagePath)
        item = QtWidgets.QTableWidgetItem(im['name'])
        rowcount = self.tableWidget.rowCount()
        self.tableWidget.insertRow(rowcount)
        imrow = rowcount
        imcol = 0
        self.tableWidget.setItem(imrow, imcol, item)
        index = self.tableWidget.indexFromItem(item)
        im['index'] = index
        self.imagePoint[index] = {"image": im,
                                  "point": {}}
        # self.tableWidget.sortItems(0)  # sort from column 0
        return index

    def addPointFile2Table(self, imageId,
                           pointFilePath: str):
        "Add point file 2 given image"
        imPoint = self.imagePoint[imageId]
        imPoint['points'] = {}
        imPoint['points']['path'] = pointFilePath
        imPoint['points']['name'] = os.path.basename(pointFilePath)
        item = QtWidgets.QTableWidgetItem(imPoint['points']['name'])
        imageItem = self.tableWidget.itemFromIndex(imageId)
        imagerow = self.tableWidget.row(imageItem)
        imagecol = self.tableWidget.column(imageItem)
        itemcol = imagecol + 1
        self.tableWidget.setItem(imagerow, itemcol, item)
        itemindex = self.tableWidget.indexFromItem(item)
        imPoint['points']['index'] = itemindex

    def importImagePoints(self):
        "Import images and points"
        # import images first then check if points exist
        self.tableWidget.clearContents()
        fdir = QtWidgets.QFileDialog.getOpenFileNames(
            self.centralwidget,
            "Select Images", "", "Images (*.png *.jpg)")
        if fdir:
            for impath in fdir[0]:
                imageId = self.addImage2Table(impath)
                pointPath = self.findPointPathFromImagePath(impath)
                if pointPath is not None:
                    self.addPointFile2Table(imageId, pointPath)

    def getImageIdFromTable(self):
        "Get selected image from table"
        rownb = self.tableWidget.currentRow()
        item = self.tableWidget.item(rownb, 0)
        if isinstance(item, int):
            if item == 0:
                return
        #
        imageId = self.tableWidget.indexFromItem(item)
        return imageId

    def importPoint(self):
        "Import point for image"
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return
        #
        fdir = QtWidgets.QFileDialog.getOpenFileNames(
            self.centralwidget,
            "Select Point Files", "", "Json files (*.json)")
        if fdir:
            for pointpath in fdir[0]:
                self.addPointFile2Table(imageId, pointpath)

    def loadImage(self):
        "Load image that is selected from table"
        # self.resetSceneState()
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return

        im = self.imagePoint[imageId]
        im = im['image']
        impath = im['path']
        self.image = Image.open(impath)
        pixmap = QtGui.QPixmap(impath)
        self.loadImage2Scene(pixmap)

    def loadImage2Scene(self, pixmap):
        self.resetSceneState()
        self.scene.image = pixmap.copy()
        self.scene.drawPixmapImage()

    def setPoints2PointEditor(self):
        "Open a point file in point editor"
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return
        impoint = self.imagePoint[imageId]
        path = impoint['points']['path']
        with open(path, 'r', encoding='utf-8', newline='\n') as f:
            points = json.load(f)
            self.pointEditor.setDataFromPointsData(points)

    def resetSceneState(self):
        self.image = None
        self.pixmapImage = QtGui.QPixmap()
        # self.scene.clear()
        self.pointEditor.clearData()

    def showGlobalThresh(self):
        if self.globalThreshCBox.isChecked():
            self.globalThreshSpinBox.show()
        else:
            self.globalThreshSpinBox.hide()

    def showGlobalCarveDir(self):
        if self.globalCarveDirCBox.isChecked():
            self.carveDirComboBox.show()
        else:
            self.carveDirComboBox.hide()

    # Standard gui
    def closeApp(self, event):
        "Close application"
        reply = QtWidgets.QMessageBox.question(
            self.centralwidget, 'Message',
            "Are you sure to quit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            sys.exit(0)
        else:
            event.ignore()
            #
        return

    def showInterface(self):
        "Show the interface"
        self.main_window.show()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindowFinal()
    window.showInterface()
    sys.exit(app.exec_())
