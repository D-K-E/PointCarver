# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE

from .. import qtapp
from ..main.pointcarver import SeamMarker
from ..main.utils import readImage, readPoints, parsePoints, stripExt
from ..main.utils import qt_image_to_array


from PIL import Image, ImageQt
from PySide2 import QtGui, QtCore, QtWidgets, QtTest
import unittest


def getComboTextContent(comboBox):
    "Get contents of a combo box"
    return [comboBox.itemText(i) for i in range(comboBox.count())]


class QtAppTest(unittest.TestCase):
    "Test your qt gui"

    def setUp(self):
        "Create your main window instance"
        self.mainwin = qtapp.AppWindowFinal()

    def test_default_values(self):
        "Test default values of the interface"
        self.assertEqual(self.mainwin.imagePoint, {})
        self.assertEqual(self.mainwin.image, None)
        self.assertEqual(self.mainwin.coords, None)
        self.assertEqual(self.mainwin.coords_colSlice, None)
        self.assertEqual(self.mainwin.coords_direction, None)
        self.assertEqual(self.mainwin.assetsdir, "")
        self.assertEqual(self.mainwin.sceneImagePoint,
                         {"image": QtGui.QPixmap(),
                          "points": []})
        self.assertEqual(self.mainwin.colors,
                         {"red": QtCore.Qt.red,
                          "black": QtCore.Qt.black,
                          "green": QtCore.Qt.green,
                          "yellow": QtCore.Qt.yellow,
                          "cyan": QtCore.Qt.cyan,
                          "blue": QtCore.Qt.blue,
                          "gray": QtCore.Qt.gray,
                          "magenta": QtCore.Qt.magenta,
                          "white": QtCore.Qt.white})
        self.assertEqual(self.mainwin.directions,
                         ['down', 'up', 'left', 'right']
                         )

        self.assertEqual(self.mainwin.scene,
                         qtapp.SceneCanvas(
                             imagePoint=self.sceneImagePoint,
                             editor=self.pointEditor,
                             pointSize=self.pointSizeSpin.value(),
                             pointColor=self.pointColorComboBox.currentText()))
        comboContents = getComboTextContent(self.mainwin.carveDirComboBox)
        self.assertEqual(comboContents,
                         ['down', 'up', 'left', 'right'])

        combovals = list(self.mainwin.colors.keys())
        comboContents = getComboTextContent(self.mainwin.pointColorComboBox)
        self.assertEqual(comboContents,
                         combovals)
        comboContents = getComboTextContent(self.mainwin.markColorComboBox)
        self.assertEqual(comboContents,
                         combovals)
        self.assertEqual(True, self.mainwin.pointEditorCbox.isChecked())
        self.assertEqual(True, self.mainwin.showTableCbox.isChecked())
