# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE

from main.pointcarver import SeamMarker
from main.utils import readImage, readPoints, parsePoints, stripExt
from ui.designerOutput import Ui_MainWindow as UIMainWindow

from PIL import Image
from PySide2 import QtGui, QtCore, QtWidgets
import sys
import os


class AppWindowInit(UIMainWindow):
    """
    Initializes the image window
    made in qt designer
    """

    def __init__(self):
        self.main_window = QtWidgets.QMainWindow()
        super().setupUi(self.main_window)
        pass


class AppWindowFinal(AppWindowInit):
    "Final application window"
    def __init__(self):
        super().__init__()
        self.imagePoint = {}  # imageId: {image: {}, points: {}}
        self.image = None
        self.scene = QtWidgets.QGraphicsScene()
        self.sceneImagePoint = {}

        # table widget related

        # Main Window Events
        self.main_window.setWindowTitle("Seam Marker using Points")
        self.main_window.closeEvent = self.closeApp

        # Buttons
        self.importBtn.clicked.connect(self.importImagePoints)

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
        assetdir = os.path.dirname(pardir)  # assetsPath
        pointsdir = os.path.join(assetdir, 'points')  # assetsPath/points
        imname = os.path.basename(imagePath)
        imNoExt = stripExt(imname)[0]
        pointsName = imNoExt + "-points.txt"
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
        item = self.tableWidget.selectedItems()
        if not item:
            self.statusbar.showMessage("Please select an image first")
            return
        elif len(item) > 1:
            self.statusbar.showMessage("Please select a single image")
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
            "Select Point Files", "", "Files (*.txt)")
        if fdir:
            for pointpath in fdir[0]:
                self.addPointFile2Table(imageId, pointpath)

    def loadImage(self):
        "Load image that is seleceted from table"
        self.sceneImagePoint = {}
        self.sceneImagePoint['image'] = {}
        self.sceneImagePoint['points'] = {}
        imageId = self.getImageIdFromTable()
        if imageId is None:
            return
        im = self.imagePoint[imageId]
        im = im['image']
        impath = im['path']
        pixmap = QtGui.QPixmap(impath)
        sceneItem = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.scene.addItem(sceneItem)










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
