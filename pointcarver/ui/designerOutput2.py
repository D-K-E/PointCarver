# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface2.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1012, 850)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.toolBox = QtWidgets.QToolBox(self.verticalLayoutWidget)
        self.toolBox.setObjectName("toolBox")
        self.tableTool = QtWidgets.QWidget()
        self.tableTool.setGeometry(QtCore.QRect(0, 0, 536, 591))
        self.tableTool.setObjectName("tableTool")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tableTool)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tableWidget = QtWidgets.QTableWidget(self.tableTool)
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setRowCount(0)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.importBtn = QtWidgets.QPushButton(self.tableTool)
        self.importBtn.setObjectName("importBtn")
        self.horizontalLayout_15.addWidget(self.importBtn)
        self.removeImagePointBtn = QtWidgets.QPushButton(self.tableTool)
        self.removeImagePointBtn.setObjectName("removeImagePointBtn")
        self.horizontalLayout_15.addWidget(self.removeImagePointBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.addImageBtn = QtWidgets.QPushButton(self.tableTool)
        self.addImageBtn.setObjectName("addImageBtn")
        self.horizontalLayout_8.addWidget(self.addImageBtn)
        self.addPoint2ImageBtn = QtWidgets.QPushButton(self.tableTool)
        self.addPoint2ImageBtn.setObjectName("addPoint2ImageBtn")
        self.horizontalLayout_8.addWidget(self.addPoint2ImageBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.loadBtn = QtWidgets.QPushButton(self.tableTool)
        self.loadBtn.setObjectName("loadBtn")
        self.horizontalLayout_10.addWidget(self.loadBtn)
        self.openPointBtn = QtWidgets.QPushButton(self.tableTool)
        self.openPointBtn.setObjectName("openPointBtn")
        self.horizontalLayout_10.addWidget(self.openPointBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.toolBox.addItem(self.tableTool, "")
        self.pointEditorTool = QtWidgets.QWidget()
        self.pointEditorTool.setGeometry(QtCore.QRect(0, 0, 536, 591))
        self.pointEditorTool.setObjectName("pointEditorTool")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.pointEditorTool)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pointsTable = QtWidgets.QTableWidget(self.pointEditorTool)
        self.pointsTable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.pointsTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pointsTable.setColumnCount(6)
        self.pointsTable.setObjectName("pointsTable")
        self.pointsTable.setRowCount(0)
        self.verticalLayout_4.addWidget(self.pointsTable)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.pointDirectionLabel = QtWidgets.QLabel(self.pointEditorTool)
        self.pointDirectionLabel.setObjectName("pointDirectionLabel")
        self.verticalLayout_15.addWidget(self.pointDirectionLabel)
        self.pointDirectionComboBox = QtWidgets.QComboBox(self.pointEditorTool)
        self.pointDirectionComboBox.setObjectName("pointDirectionComboBox")
        self.verticalLayout_15.addWidget(self.pointDirectionComboBox)
        self.horizontalLayout_11.addLayout(self.verticalLayout_15)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.thresholdLabel = QtWidgets.QLabel(self.pointEditorTool)
        self.thresholdLabel.setObjectName("thresholdLabel")
        self.verticalLayout_12.addWidget(self.thresholdLabel)
        self.thresholdSpin = QtWidgets.QSpinBox(self.pointEditorTool)
        self.thresholdSpin.setMinimum(1)
        self.thresholdSpin.setObjectName("thresholdSpin")
        self.verticalLayout_12.addWidget(self.thresholdSpin)
        self.horizontalLayout_11.addLayout(self.verticalLayout_12)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.pointSizeLabel = QtWidgets.QLabel(self.pointEditorTool)
        self.pointSizeLabel.setObjectName("pointSizeLabel")
        self.verticalLayout_9.addWidget(self.pointSizeLabel)
        self.pointSizeSpin = QtWidgets.QSpinBox(self.pointEditorTool)
        self.pointSizeSpin.setMinimum(1)
        self.pointSizeSpin.setMaximum(10)
        self.pointSizeSpin.setObjectName("pointSizeSpin")
        self.verticalLayout_9.addWidget(self.pointSizeSpin)
        self.horizontalLayout_5.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.pointColorLabel = QtWidgets.QLabel(self.pointEditorTool)
        self.pointColorLabel.setObjectName("pointColorLabel")
        self.verticalLayout_8.addWidget(self.pointColorLabel)
        self.pointColorComboBox = QtWidgets.QComboBox(self.pointEditorTool)
        self.pointColorComboBox.setObjectName("pointColorComboBox")
        self.verticalLayout_8.addWidget(self.pointColorComboBox)
        self.horizontalLayout_5.addLayout(self.verticalLayout_8)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.pointXLabel = QtWidgets.QLabel(self.pointEditorTool)
        self.pointXLabel.setObjectName("pointXLabel")
        self.horizontalLayout_13.addWidget(self.pointXLabel)
        self.pointYLabel = QtWidgets.QLabel(self.pointEditorTool)
        self.pointYLabel.setObjectName("pointYLabel")
        self.horizontalLayout_13.addWidget(self.pointYLabel)
        self.verticalLayout_7.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.pointXSpinBox = QtWidgets.QSpinBox(self.pointEditorTool)
        self.pointXSpinBox.setObjectName("pointXSpinBox")
        self.horizontalLayout_14.addWidget(self.pointXSpinBox)
        self.pointYSpinBox = QtWidgets.QSpinBox(self.pointEditorTool)
        self.pointYSpinBox.setObjectName("pointYSpinBox")
        self.horizontalLayout_14.addWidget(self.pointYSpinBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_5.addLayout(self.verticalLayout_7)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4.addLayout(self.verticalLayout_6)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.drawPointsBtn = QtWidgets.QPushButton(self.pointEditorTool)
        self.drawPointsBtn.setObjectName("drawPointsBtn")
        self.horizontalLayout_9.addWidget(self.drawPointsBtn)
        self.savePointsBtn = QtWidgets.QPushButton(self.pointEditorTool)
        self.savePointsBtn.setObjectName("savePointsBtn")
        self.horizontalLayout_9.addWidget(self.savePointsBtn)
        self.deletePointBtn = QtWidgets.QPushButton(self.pointEditorTool)
        self.deletePointBtn.setObjectName("deletePointBtn")
        self.horizontalLayout_9.addWidget(self.deletePointBtn)
        self.verticalLayout_4.addLayout(self.horizontalLayout_9)
        self.toolBox.addItem(self.pointEditorTool, "")
        self.verticalLayout_5.addWidget(self.toolBox)
        self.carveGroup = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.carveGroup.setObjectName("carveGroup")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.carveGroup)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.saveCoordinatesBtn = QtWidgets.QPushButton(self.carveGroup)
        self.saveCoordinatesBtn.setObjectName("saveCoordinatesBtn")
        self.horizontalLayout_7.addWidget(self.saveCoordinatesBtn)
        self.saveBtn = QtWidgets.QPushButton(self.carveGroup)
        self.saveBtn.setObjectName("saveBtn")
        self.horizontalLayout_7.addWidget(self.saveBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.carveBtn = QtWidgets.QPushButton(self.carveGroup)
        self.carveBtn.setObjectName("carveBtn")
        self.horizontalLayout_3.addWidget(self.carveBtn)
        self.saveAllBtn = QtWidgets.QPushButton(self.carveGroup)
        self.saveAllBtn.setObjectName("saveAllBtn")
        self.horizontalLayout_3.addWidget(self.saveAllBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_5.addWidget(self.carveGroup)
        self.canvas = QtWidgets.QGraphicsView(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.canvas.sizePolicy().hasHeightForWidth())
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.setObjectName("canvas")
        self.verticalLayout_10.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1012, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.importBtn.setText(_translate("MainWindow", "Import Files"))
        self.removeImagePointBtn.setText(_translate("MainWindow", "Remove Selections"))
        self.addImageBtn.setText(_translate("MainWindow", "Add Image"))
        self.addPoint2ImageBtn.setText(_translate("MainWindow", "Add Point File to Image"))
        self.loadBtn.setText(_translate("MainWindow", "Load Image"))
        self.openPointBtn.setToolTip(_translate("MainWindow", "Opens the point file in Point Editor"))
        self.openPointBtn.setText(_translate("MainWindow", "Open Point File"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.tableTool), _translate("MainWindow", "Image Point Table"))
        self.pointsTable.setSortingEnabled(True)
        self.pointDirectionLabel.setText(_translate("MainWindow", "Carve Direction"))
        self.thresholdLabel.setText(_translate("MainWindow", "Threshold"))
        self.pointSizeLabel.setText(_translate("MainWindow", "Point Size"))
        self.pointColorLabel.setText(_translate("MainWindow", "Point Color"))
        self.pointXLabel.setText(_translate("MainWindow", "X"))
        self.pointYLabel.setText(_translate("MainWindow", "Y"))
        self.drawPointsBtn.setText(_translate("MainWindow", "Draw Points"))
        self.savePointsBtn.setText(_translate("MainWindow", "Save Points"))
        self.deletePointBtn.setText(_translate("MainWindow", "Delete Point"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.pointEditorTool), _translate("MainWindow", "Point Editor"))
        self.saveCoordinatesBtn.setText(_translate("MainWindow", "Save Coordinates"))
        self.saveBtn.setText(_translate("MainWindow", "Save Segments"))
        self.carveBtn.setText(_translate("MainWindow", "Carve"))
        self.saveAllBtn.setText(_translate("MainWindow", "Save All"))

