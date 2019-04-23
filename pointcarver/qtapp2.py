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
                 tableWidget: QtWidgets.QTableWidget,
                 directionWidget: QtWidgets.QComboBox,
                 thresholdWidget: QtWidgets.QSpinBox,
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
        self.table = tableWidget
        self.directionWidget = directionWidget
        self.directionWidget.setEditable(False)
        self.directionWidget.setDuplicatesEnabled(False)

        self.thresholdWidget = thresholdWidget
        self.colorWidget = colorWidget
        self.sizeWidget = sizeWidget
        self.xValSpin = xValueWidget
        self.yValSpin = yValueWidget
        pointDataTemplate = {'row index':
                             {"coord": {'val': (0, 0), 'index': 'tableIndex'},
                              "color": {"val": '', 'index': 'tableIndex'},
                              "direction": {"val": "", 'index': 'tableIndex'},
                              "threshold": {"val": 1, 'index': 'tableIndex'},
                              "size": {"val": 1, 'index': 'tableIndex'},
                              "x": {"val": 0, 'index': 'tableIndex'},
                              "y": {"val": 0, 'index': 'tableIndex'},
                              }
                             }
        self.pointsData = {}

        self.colors = colors
        super().__init__(parent=parent)

        # Events
        self.xValSpin.valueChanged.connect(self.setXval2Point)
        self.yValSpin.valueChanged.connect(self.setYval2Point)
        self.thresholdWidget.valueChanged.connect(self.setThreshold2Point)
        self.colorWidget.currentTextChanged.connect(
            self.setColor2Point)
        self.directionWidget.currentTextChanged.connect(
            self.setDirection2Point)
        self.sizeWidget.valueChanged.connect(self.setSize2Point)
        self.table.itemSelectionChanged.connect(self.showRowData)

    def getRowData(self):
        "Get current coords"
        # x, y
        row = self.table.currentRow()
        data = self.pointsData.get(row)
        return row, data

    def getPointDataFromWidgets(self):
        "set points and related data to widgets"
        color = self.colorWidget.currentText()
        pointSize = self.sizeWidget.value()
        direction = self.directionWidget.currentText()
        threshold = self.thresholdWidget.value()
        return {"color": color, "size": pointSize,
                "direction": direction, "threshold": threshold}

    def setThreshold2Point(self):
        row, data = self.getRowData()
        if data is None:
            return
        newval = self.thresholdWidget.value()
        data['threshold'] = newval
        self.setData2Row(row, data)

    def setDirection2Point(self):
        row, data = self.getRowData()
        if data is None:
            return
        newval = self.directionWidget.currentText()
        data['direction'] = newval
        self.setData2Row(row, data)

    def setColor2Point(self):
        row, data = self.getRowData()
        if data is None:
            return
        newval = self.colorWidget.currentText()
        data['color'] = newval
        self.setData2Row(row, data)

    def setSize2Point(self):
        row, data = self.getRowData()
        if data is None:
            return
        newval = self.sizeWidget.value()
        data['size'] = newval
        self.setData2Row(row, data)

    def setXval2Point(self):
        row, data = self.getRowData()
        if data is None:
            return
        newval = self.xValSpin.value()
        data['x'] = newval
        coords = list(data['coordinates'])
        coords[0] = newval
        data['coordinates'] = tuple(coords)
        self.setData2Row(row, data)

    def setYval2Point(self):
        row, data = self.getRowData()
        if data is None:
            return
        newval = self.yValSpin.value()
        data['y'] = newval
        coords = list(data['coordinates'])
        coords[1] = newval
        data['coordinates'] = tuple(coords)
        self.setData2Row(row, data)

    def showRowData(self):
        "Show point data on widgets"
        row, data = self.getRowData()
        if data is None:
            return
        color = data['color']
        direction = data['direction']
        threshold = data['threshold']
        size = data['size']
        x = data['x']
        y = data['y']
        # self.coordsCombo.setCurrentText(str(pointCoords))
        self.directionWidget.setCurrentText(direction)
        self.colorWidget.setCurrentText(color)
        self.thresholdWidget.setValue(threshold)
        self.xValSpin.setValue(x)
        self.yValSpin.setValue(y)
        self.sizeWidget.setValue(size)

    def convertData2TableWidgetItems(self, data):
        "Converts dict data to table widget items to add to table widget"
        coordsItem = QtWidgets.QTableWidgetItem(str(data['coordinates']))
        xItem = QtWidgets.QTableWidgetItem(str(data['x']))
        yItem = QtWidgets.QTableWidgetItem(str(data['y']))
        threshItem = QtWidgets.QTableWidgetItem(str(data['threshold']))
        directionItem = QtWidgets.QTableWidgetItem(data['direction'])
        colorItem = QtWidgets.QTableWidgetItem(data['color'])
        sizeItem = QtWidgets.QTableWidgetItem(str(data['size']))
        return [coordsItem, xItem, yItem, directionItem,
                threshItem, sizeItem, colorItem]

    def setData2Row(self, rownb: int, data: dict):
        "Set data to given row"
        self.pointsData[rownb] = data
        itemList = self.convertData2TableWidgetItems(data)
        for i in range(len(itemList)):
            newitem = itemList[i]
            newitem.setFlags(QtCore.Qt.ItemIsEditable)
            newitem.setFlags(QtCore.Qt.ItemIsSelectable)
            self.table.setItem(rownb, i, newitem)

    def addPointCoordsWithCurrentData(self,
                                      pointCoords: (int, int)):
        "Set point data from points data by using coords"
        rowcount = self.table.rowCount()
        self.table.insertRow(rowcount)
        data = self.getPointDataFromWidgets()
        data['x'] = pointCoords[0]
        data['y'] = pointCoords[1]
        data['coordinates'] = pointCoords
        self.setData2Row(rowcount, data)

    def setPointsData2Table(self, pointsData: dict):
        "Set widgets' data from points data"
        self.clearData()
        for row, data in pointsData.items():
            data['coordinates'] = (data['x'], data['y'])
            rownb = int(row)
            self.table.insertRow(rownb)
            self.setData2Row(rownb, data)

    def setTableData2PointsData(self):
        rowcount = self.table.rowCount()
        self.pointsData = {}
        for r in range(rowcount):
            self.pointsData[r] = {}
            # coords
            item = self.table.itemAt(r, 0)
            item = item.text()
            if item != '':
                item = item.strip('()')
                item = tuple([int(i) for i in item.split(',') if i])
                self.pointsData[r]['coordinates'] = item
            # x, y
            item = self.table.itemAt(r, 1)
            item = item.text()
            item = int(item)
            self.pointsData[r]['x'] = item
            item = self.table.itemAt(r, 2)
            item = item.text()
            item = int(item)
            self.pointsData[r]['y'] = item
            # direction
            item = self.table.itemAt(r, 3)
            item = item.text()
            self.pointsData[r]['direction'] = item
            # threshold
            item = self.table.itemAt(r, 4)
            item = item.text()
            item = int(item)
            self.pointsData[r]['threshold'] = item
            # size
            item = self.table.itemAt(r, 5)
            item = item.text()
            item = int(item)
            self.pointsData[r]['size'] = item
            # color
            item = self.table.itemAt(r, 6)
            item = item.text()
            self.pointsData[r]['color'] = item

    def removeRow(self):
        "remove selected rows from table and points data"
        row, data = self.getRowData()
        if data is None:
            return
        self.table.removeRow(row)
        self.setTableData2PointsData()

    def clearData(self):
        self.pointsData = {}
        self.table.clearContents()


class SceneCanvas(QtWidgets.QGraphicsScene):
    "Mouse events overriding for graphics scene with editor integration"

    def __init__(self,
                 pointEditor: PointEditor,
                 image: QtGui.QPixmap,
                 parent=None):
        self.pointEditor = pointEditor
        self.image = image
        if image:
            imw = image.width()
            imh = image.height()
            pointEditor.xValSpin.setMaximum(imw)
            pointEditor.xValSpin.setMinimum(0)
            pointEditor.yValSpin.setMaximum(imh)
            pointEditor.yValSpin.setMinimum(0)
        super().__init__(parent)

    def addPoint2PointEditor(self, point):
        "Add point coords to point editor"
        self.pointEditor.addPointCoordsWithCurrentData(point)

    def drawPointsOnImage(self):
        self.clear()
        image = self.image.copy()
        imw, imh = image.width(), image.height()
        result = QtGui.QPixmap(w=imw, h=imh)
        result.fill(QtCore.Qt.white)
        painter = QtGui.QPainter()
        painter.begin(image)
        painter.drawPixmap(0, 0, image)
        # pdb.set_trace()
        for row, data in self.pointEditor.pointsData.items():
            if data['color'] == '':
                color = QtCore.Qt.red
            else:
                color = self.pointEditor.colors[data['color']]
            size = data['size']
            brush = QtGui.QBrush(color)
            painter.setBrush(brush)
            painter.setPen(color)
            px = data['x']
            py = data['y']
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
        point = (int(x), int(y))
        self.addPoint2PointEditor(point)
        self.drawPointsOnImage()


class AppWindowFinal(AppWindowInit):
    "Application window"

    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmapImage = QtGui.QPixmap()
        self.imageId = None
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
        self.pointColorComboBox.addItems(combovals)
        self.pointColorComboBox.setCurrentText("red")
        self.directions = ['down', 'up', 'left', 'right']
        self.pointDirectionComboBox.addItems(self.directions)
        self.pointDirectionComboBox.setCurrentText("down")
        self.carveDirComboBox.addItems(self.directions)
        pointProperties = ['coordinates', 'x', 'y', 'direction', 'threshold',
                           'size', 'color']
        self.pointsTable.setColumnCount(len(pointProperties))
        self.pointsTable.setHorizontalHeaderLabels(
            pointProperties
        )
        self.pointEditor = PointEditor(
            tableWidget=self.pointsTable,
            directionWidget=self.pointDirectionComboBox,
            thresholdWidget=self.thresholdSpin,
            colorWidget=self.pointColorComboBox,
            xValueWidget=self.pointXSpinBox,
            yValueWidget=self.pointYSpinBox,
            sizeWidget=self.pointSizeSpin,
        )
        self.scene = SceneCanvas(self.pointEditor,
                                 self.image)
        self.toolBox.setCurrentIndex(0)

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
        self.drawPointsBtn.clicked.connect(self.drawEditorPoints)
        # self.carveBtn.clicked.connect(self.markSeamsOnImage)
        self.savePointsBtn.clicked.connect(self.savePoints)
        # self.saveCoordinatesBtn.clicked.connect(self.saveSeamCoordinates)
        # self.saveBtn.clicked.connect(self.saveSegments)
        # self.saveAllBtn.clicked.connect(self.saveAll)
        self.addPoint2ImageBtn.clicked.connect(self.importPoint)
        self.openPointBtn.clicked.connect(self.setPoints2PointEditor)
        self.deletePointBtn.clicked.connect(self.pointEditor.removeRow)

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
        self.resetSceneState()
        self.pointEditor.clearData()
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return
        self.imageId = imageId
        im = self.imagePoint[imageId]
        im = im['image']
        impath = im['path']
        # pdb.set_trace()
        self.image = Image.open(impath)
        pixmap = QtGui.QPixmap(impath)
        self.loadImage2Scene(pixmap)

    def loadImage2Scene(self, pixmap):
        self.resetSceneState()
        self.pixmapImage = pixmap.copy()
        self.renderScenePoints()

    def checkPointData(self):
        "Check point data"
        if 0 in self.pointEditor.pointsData.keys():
            return True
        else:
            return False

    def renderScenePoints(self):
        self.scene = SceneCanvas(pointEditor=self.pointEditor,
                                 image=self.pixmapImage)
        # pdb.set_trace()
        if self.checkPointData() is True:
            self.scene.drawPointsOnImage()
        else:
            self.scene.drawPixmapImage()
        #
        self.canvas.setScene(self.scene)
        self.canvas.setCursor(QtCore.Qt.CrossCursor)
        self.canvas.show()

    def drawEditorPoints(self):
        self.renderScenePoints()

    def getImageNameFromId(self):
        "Get image name from image id without extension"
        imageId = self.imageId
        im = self.imagePoint[imageId]
        im = im['image']
        imname = im['name']
        imname, ext = stripExt(imname)
        return imname

    def savePoints(self):
        "Save points to system"
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
                json.dump(self.pointEditor.pointsData, 
                          f, ensure_ascii=False, indent=2)

    def setPoints2PointEditor(self):
        "Open a point file in point editor"
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return
        impoint = self.imagePoint[imageId]
        path = impoint['points']['path']
        with open(path, 'r', encoding='utf-8', newline='\n') as f:
            pointsData = json.load(f)
            self.pointEditor.setPointsData2Table(pointsData)

    def resetSceneState(self):
        self.image = None
        self.pixmapImage = QtGui.QPixmap()
        self.scene.clear()
        # self.pointEditor.clearData()

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
