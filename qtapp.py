# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE

from main.pointcarver import SeamMarker
from main.utils import readImage, readPoints, parsePoints, stripExt
from main.utils import qt_image_to_array
from ui.designerOutput import Ui_MainWindow as UIMainWindow

from PIL import Image, ImageQt
from PySide2 import QtGui, QtCore, QtWidgets
import sys
import os
import json
import numpy as np


class AppWindowInit(UIMainWindow):
    """
    Initializes the image window
    made in qt designer
    """

    def __init__(self):
        self.main_window = QtWidgets.QMainWindow()
        super().setupUi(self.main_window)
        pass


class SceneCanvas(QtWidgets.QGraphicsScene):
    "Mouse events overriding for graphics scene with editor integration"

    def __init__(self,
                 imagePoint,
                 pointSize: int,
                 editor: QtWidgets.QPlainTextEdit,
                 parent=None,
                 pointColor=QtCore.Qt.green
                 ):
        self.editor = editor
        self.imagePoint = imagePoint
        self.pointColor = pointColor
        self.pointSize = pointSize
        super().__init__(parent)

    def addPoints2ImagePoint(self, point):
        "Add points 2 image point attribute"
        self.imagePoint['points'].append(point)

    def renderPointsInEditor(self):
        "Render points of imagePoint in pointEditor"
        self.editor.clear()
        pointjson = [{'y': int(p[0]),
                      'x': int(p[1])} for p in self.imagePoint['points']]
        pointjson = json.dumps(pointjson, ensure_ascii=False, indent=2)
        self.editor.setPlainText(pointjson)

    def drawPointsOnImage(self):
        "Draw points as an overlay on the image"
        self.clear()
        points = self.imagePoint['points']
        image = self.imagePoint['image']
        imw, imh = image.width(), image.height()
        brush = QtGui.QBrush(self.pointColor)
        pwidth = self.pointSize
        pheight = self.pointSize
        result = QtGui.QPixmap(w=imw, h=imh)
        result.fill(QtCore.Qt.white)
        print("isnull image: ", str(image.isNull()))
        painter = QtGui.QPainter()
        painter.begin(image)
        painter.setBrush(brush)
        painter.setPen(self.pointColor)
        painter.drawPixmap(0, 0, image)

        for point in points:
            py = point[0]
            px = point[1]
            painter.drawEllipse(px, py, pwidth, pheight)
        #
        painter.end()
        pixmapItem = QtWidgets.QGraphicsPixmapItem(image)
        self.addItem(pixmapItem)

    def mouseDoubleClickEvent(self, event):
        "Overriding double click event"
        point = event.scenePos()
        x, y = point.x(), point.y()
        print(x, " ", y)
        pointInv = [int(y), int(x)]
        self.addPoints2ImagePoint(pointInv)
        self.renderPointsInEditor()
        self.drawPointsOnImage()


class AppWindowFinal(AppWindowInit):
    "Final application window"

    def __init__(self):
        super().__init__()
        self.imagePoint = {}  # imageId: {image: {}, points: {}}
        self.image = None
        self.assetsdir = ""
        self.sceneImagePoint = {"image": QtGui.QPixmap(),
                                "points": []}
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
        self.pointColorComboBox.addItems(combovals)
        self.markColorComboBox.addItems(combovals)
        self.scene = SceneCanvas(
            imagePoint=self.sceneImagePoint,
            editor=self.pointEditor,
            pointSize=self.pointSizeSpin.value(),
            pointColor=self.pointColorComboBox.currentText())
        self.directions = ['down', 'up', 'left', 'right']
        self.carveDirComboBox.addItems(self.directions)

        # table widget related

        # Main Window Events
        self.main_window.setWindowTitle("Seam Marker using Points")
        self.main_window.closeEvent = self.closeApp

        # Point size spin
        self.pointSizeSpin.valueChanged.connect(self.setPointSize2Scene)

        # Point color Combo
        self.pointColorComboBox.currentTextChanged.connect(
            self.setPointColor2Scene)

        # Buttons
        self.importBtn.clicked.connect(self.importImagePoints)
        self.loadBtn.clicked.connect(self.loadImage)
        self.drawPointsBtn.clicked.connect(self.drawEditorPoints)
        self.carveBtn.clicked.connect(self.markSeamsOnImage)
        self.savePointsBtn.clicked.connect(self.savePoints)
        self.saveCoordinatesBtn.clicked.connect(self.saveSeamCoordinates)
        self.addPoint2ImageBtn.clicked.connect(self.importPoint)
        self.openPointBtn.clicked.connect(self.openPointInEditor)

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
        self.tableWidget.sortItems(0)  # sort from column 0
        return index

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

    def importImagePoints(self):
        "Import images and points"
        # import images first then check if points exist
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
        items = self.tableWidget.selectedItems()
        if not items:
            self.statusbar.showMessage("Please select an image first")
            return
        elif len(items) > 1:
            self.statusbar.showMessage("Please select a single image")
            return
        #
        imageId = self.tableWidget.indexFromItem(items[0])
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

    def openPointInEditor(self):
        "Open a point file in point editor"
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return
        impoint = self.imagePoint[imageId]
        path = impoint['points']['path']
        with open(path, 'r', encoding='utf-8', newline='\n') as f:
            jobjstr = f.read()
            self.pointEditor.setPlainText(jobjstr)

    def getImageNameFromId(self):
        "Get image name from image id without extension"
        imageId = self.sceneImagePoint['imageListId']
        im = self.imagePoint[imageId]
        im = im['image']
        imname = im['name']
        imname, ext = stripExt(imname)
        return imname

    def savePoints(self):
        "Save points in the editor from file dialog"
        text = self.pointEditor.toPlainText()
        imname = self.getImageNameFromId()
        pointsname = imname + "-points.json"
        path = os.path.join(self.assetsdir, pointsname)
        fileName = QtWidgets.QFileDialog.getSaveFileName(self.centralwidget,
                                                         "Save Point File",
                                                         path,
                                                         'Json Files (*.json)')
        fpath = fileName[0]
        if fpath:
            with open(fpath, "w", encoding='utf-8', newline='\n') as f:
                jtext = json.loads(text)
                json.dump(jtext, f, ensure_ascii=False, indent=2)

    def loadImage(self):
        "Load image that is selected from table"
        self.sceneImagePoint = {}
        self.sceneImagePoint['points'] = []
        self.pointEditor.clear()
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return

        im = self.imagePoint[imageId]
        im = im['image']
        impath = im['path']
        self.sceneImagePoint['imageListId'] = imageId
        self.image = Image.open(impath)
        pixmap = QtGui.QPixmap(impath)
        self.loadImage2Scene(pixmap)

    def loadImage2Scene(self, pixmap):
        "Load a given image pixmap to scene"
        sceneItem = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.sceneImagePoint['image'] = pixmap
        self.renderSceneImagePoint()

    def getPointColor(self):
        "Get point color from combo box"
        color = self.pointColorComboBox.currentText()
        return self.colors[color]

    def setPointColor2Scene(self):
        "Set point color to scene"
        color = self.pointColorComboBox.currentText()
        pointColor = self.colors[color]
        self.scene.pointColor = pointColor

    def setPointSize2Scene(self):
        "Set point size to scene widget"
        size = self.pointSizeSpin.value()
        self.scene.pointSize = size

    def drawEditorPoints(self):
        "Draw points on editor to scene"
        self.resetSceneImage()
        pointstr = self.pointEditor.toPlainText()
        pointdict = json.loads(pointstr)
        points = [[int(p['y']), int(p['x'])] for p in pointdict]
        self.sceneImagePoint['points'] = points
        self.renderSceneImagePoint()

    def renderSceneImagePoint(self):
        "render image point object to scene"
        self.scene.clear()
        imagePixmap = self.sceneImagePoint['image']
        imageItem = QtWidgets.QGraphicsPixmapItem(imagePixmap)
        points = self.sceneImagePoint['points']
        pointSize = self.pointSizeSpin.value()
        pointColor = self.getPointColor()
        self.scene = SceneCanvas(imagePoint=self.sceneImagePoint,
                                 editor=self.pointEditor,
                                 pointColor=pointColor,
                                 pointSize=pointSize)
        self.scene.drawPointsOnImage()
        self.canvas.setScene(self.scene)
        self.canvas.show()

    def getMarkerParams(self):
        "Get seam marker parameters from ui"
        points = self.sceneImagePoint['points']
        image = self.sceneImagePoint['image']
        rgb32image = image.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        imarr = qt_image_to_array(rgb32image)
        imarr = imarr.astype(np.uint8)
        if imarr.shape[2] > 3:
            imarr = imarr[:, :, :3]
        thresh = self.thresholdSpin.value()
        direction = self.carveDirComboBox.currentText()
        markColor = self.markColorComboBox.currentText()  # gives text val
        markColor = self.colors[markColor]  # gives a QGlobalColor object
        markColor = QtGui.QColor(markColor)
        markColor = list(markColor.getRgb()[:3])
        return imarr, points, thresh, direction, markColor

    def markSeamsOnImage(self):
        "Given point list and an image mark seam lines on the image"
        self.resetSceneImage()
        imarr, points, thresh, direction, markColor = self.getMarkerParams()
        marker = SeamMarker(imarr, points)
        markedImage = marker.markPointListSeam(imarr,
                                               points,
                                               direction,
                                               thresh,
                                               markColor)
        pilim = Image.fromarray(markedImage)
        qtimg = ImageQt.ImageQt(pilim)
        pixmap = QtGui.QPixmap.fromImage(qtimg)
        self.sceneImagePoint['image'] = pixmap
        self.renderSceneImagePoint()

    def getSeamCoordinates(self):
        "Get seam coordinates from image"
        imarr, points, thresh, direction, markColor = self.getMarkerParams()
        marker = SeamMarker(imarr, points)
        coords = marker.getPointListSeamCoordinate(
            imarr, points, direction, thresh, markColor)
        return coords

    def saveSeamCoordinates(self):
        "Save seam coordinates to a file location"
        coords = self.getSeamCoordinates()
        for coord in coords:
            marks = coord['markCoordinates']
            marks = marks.tolist()
            coord['markCoordinates'] = marks
        #
        imname = self.getImageNameFromId()
        path = os.path.join(self.assetsdir, imname + "-coordinates.json")
        fileName = QtWidgets.QFileDialog.getSaveFileName(self.centralwidget,
                                                         "Save Mark Coordinates",
                                                         path,
                                                         'Json Files (*.json)')
        fpath = fileName[0]
        if fpath:
            with open(fpath, "w", encoding='utf-8', newline='\n') as f:
                json.dump(coords, f, ensure_ascii=False, indent=2)

    def resetSceneImage(self):
        "Deletes the carves on the image"
        qtimg = ImageQt.ImageQt(self.image.copy())
        pixmap = QtGui.QPixmap.fromImage(qtimg)
        self.sceneImagePoint['image'] = pixmap
        self.loadImage2Scene(pixmap)

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
