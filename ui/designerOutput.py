# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/interface.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1006, 840)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.showTableCbox = QtWidgets.QCheckBox(self.centralwidget)
        self.showTableCbox.setObjectName("showTableCbox")
        self.verticalLayout_5.addWidget(self.showTableCbox)
        self.tableGroup = QtWidgets.QGroupBox(self.centralwidget)
        self.tableGroup.setObjectName("tableGroup")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.tableGroup)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.tableWidget = QtWidgets.QTableWidget(self.tableGroup)
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setRowCount(0)
        self.verticalLayout_7.addWidget(self.tableWidget)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.importBtn = QtWidgets.QPushButton(self.tableGroup)
        self.importBtn.setObjectName("importBtn")
        self.horizontalLayout_8.addWidget(self.importBtn)
        self.addPoint2ImageBtn = QtWidgets.QPushButton(self.tableGroup)
        self.addPoint2ImageBtn.setObjectName("addPoint2ImageBtn")
        self.horizontalLayout_8.addWidget(self.addPoint2ImageBtn)
        self.verticalLayout_7.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.loadBtn = QtWidgets.QPushButton(self.tableGroup)
        self.loadBtn.setObjectName("loadBtn")
        self.horizontalLayout_10.addWidget(self.loadBtn)
        self.openPointBtn = QtWidgets.QPushButton(self.tableGroup)
        self.openPointBtn.setObjectName("openPointBtn")
        self.horizontalLayout_10.addWidget(self.openPointBtn)
        self.verticalLayout_7.addLayout(self.horizontalLayout_10)
        self.verticalLayout_5.addWidget(self.tableGroup)
        self.pointEditorCbox = QtWidgets.QCheckBox(self.centralwidget)
        self.pointEditorCbox.setObjectName("pointEditorCbox")
        self.verticalLayout_5.addWidget(self.pointEditorCbox)
        self.pointEditorGroup = QtWidgets.QGroupBox(self.centralwidget)
        self.pointEditorGroup.setObjectName("pointEditorGroup")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.pointEditorGroup)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.editPointsCbox = QtWidgets.QCheckBox(self.pointEditorGroup)
        self.editPointsCbox.setObjectName("editPointsCbox")
        self.verticalLayout_4.addWidget(self.editPointsCbox)
        self.pointEditor = QtWidgets.QPlainTextEdit(self.pointEditorGroup)
        self.pointEditor.setObjectName("pointEditor")
        self.verticalLayout_4.addWidget(self.pointEditor)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.pointSizeLabel = QtWidgets.QLabel(self.pointEditorGroup)
        self.pointSizeLabel.setObjectName("pointSizeLabel")
        self.verticalLayout_9.addWidget(self.pointSizeLabel)
        self.pointSizeSpin = QtWidgets.QSpinBox(self.pointEditorGroup)
        self.pointSizeSpin.setMinimum(1)
        self.pointSizeSpin.setMaximum(10)
        self.pointSizeSpin.setObjectName("pointSizeSpin")
        self.verticalLayout_9.addWidget(self.pointSizeSpin)
        self.horizontalLayout_5.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.pointColorLabel = QtWidgets.QLabel(self.pointEditorGroup)
        self.pointColorLabel.setObjectName("pointColorLabel")
        self.verticalLayout_8.addWidget(self.pointColorLabel)
        self.pointColorComboBox = QtWidgets.QComboBox(self.pointEditorGroup)
        self.pointColorComboBox.setObjectName("pointColorComboBox")
        self.verticalLayout_8.addWidget(self.pointColorComboBox)
        self.horizontalLayout_5.addLayout(self.verticalLayout_8)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4.addLayout(self.verticalLayout_6)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.drawPointsBtn = QtWidgets.QPushButton(self.pointEditorGroup)
        self.drawPointsBtn.setObjectName("drawPointsBtn")
        self.horizontalLayout_9.addWidget(self.drawPointsBtn)
        self.savePointsBtn = QtWidgets.QPushButton(self.pointEditorGroup)
        self.savePointsBtn.setObjectName("savePointsBtn")
        self.horizontalLayout_9.addWidget(self.savePointsBtn)
        self.verticalLayout_4.addLayout(self.horizontalLayout_9)
        self.verticalLayout_5.addWidget(self.pointEditorGroup)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.thresholdLabel = QtWidgets.QLabel(self.centralwidget)
        self.thresholdLabel.setObjectName("thresholdLabel")
        self.verticalLayout_12.addWidget(self.thresholdLabel)
        self.thresholdSpin = QtWidgets.QSpinBox(self.centralwidget)
        self.thresholdSpin.setMinimum(1)
        self.thresholdSpin.setObjectName("thresholdSpin")
        self.verticalLayout_12.addWidget(self.thresholdSpin)
        self.horizontalLayout_6.addLayout(self.verticalLayout_12)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.markColorLabel = QtWidgets.QLabel(self.centralwidget)
        self.markColorLabel.setObjectName("markColorLabel")
        self.verticalLayout_10.addWidget(self.markColorLabel)
        self.markColorComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.markColorComboBox.setObjectName("markColorComboBox")
        self.verticalLayout_10.addWidget(self.markColorComboBox)
        self.horizontalLayout_6.addLayout(self.verticalLayout_10)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.carveDirLabel = QtWidgets.QLabel(self.centralwidget)
        self.carveDirLabel.setObjectName("carveDirLabel")
        self.verticalLayout_11.addWidget(self.carveDirLabel)
        self.carveDirComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.carveDirComboBox.setObjectName("carveDirComboBox")
        self.verticalLayout_11.addWidget(self.carveDirComboBox)
        self.horizontalLayout_6.addLayout(self.verticalLayout_11)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.saveAllBtn = QtWidgets.QPushButton(self.centralwidget)
        self.saveAllBtn.setObjectName("saveAllBtn")
        self.horizontalLayout_4.addWidget(self.saveAllBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.saveCoordinatesBtn = QtWidgets.QPushButton(self.centralwidget)
        self.saveCoordinatesBtn.setObjectName("saveCoordinatesBtn")
        self.horizontalLayout_7.addWidget(self.saveCoordinatesBtn)
        self.saveBtn = QtWidgets.QPushButton(self.centralwidget)
        self.saveBtn.setObjectName("saveBtn")
        self.horizontalLayout_7.addWidget(self.saveBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.carveBtn = QtWidgets.QPushButton(self.centralwidget)
        self.carveBtn.setObjectName("carveBtn")
        self.horizontalLayout_3.addWidget(self.carveBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_5.addLayout(self.verticalLayout_3)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.canvas = QtWidgets.QGraphicsView(self.centralwidget)
        self.canvas.setObjectName("canvas")
        self.horizontalLayout.addWidget(self.canvas)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_2.setStretch(0, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1006, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.showTableCbox.setText(_translate("MainWindow", "Show Table"))
        self.tableGroup.setTitle(_translate("MainWindow", "Image Point Table"))
        self.importBtn.setText(_translate("MainWindow", "Import Files"))
        self.addPoint2ImageBtn.setText(_translate("MainWindow", "Add Point File to Image"))
        self.loadBtn.setText(_translate("MainWindow", "Load Image"))
        self.openPointBtn.setToolTip(_translate("MainWindow", "Opens the point file in Point Editor"))
        self.openPointBtn.setText(_translate("MainWindow", "Open Point File"))
        self.pointEditorCbox.setText(_translate("MainWindow", "Show Point Editor"))
        self.pointEditorGroup.setTitle(_translate("MainWindow", "Point Editor"))
        self.editPointsCbox.setText(_translate("MainWindow", "Edit Points"))
        self.pointSizeLabel.setText(_translate("MainWindow", "Point Size"))
        self.pointColorLabel.setText(_translate("MainWindow", "Point Color"))
        self.drawPointsBtn.setText(_translate("MainWindow", "Draw Points"))
        self.savePointsBtn.setText(_translate("MainWindow", "Save Points"))
        self.thresholdLabel.setText(_translate("MainWindow", "Threshold"))
        self.markColorLabel.setText(_translate("MainWindow", "Mark Color"))
        self.carveDirLabel.setText(_translate("MainWindow", "Carve Direction"))
        self.saveAllBtn.setText(_translate("MainWindow", "Save All"))
        self.saveCoordinatesBtn.setText(_translate("MainWindow", "Save Coordinates"))
        self.saveBtn.setText(_translate("MainWindow", "Save Segments"))
        self.carveBtn.setText(_translate("MainWindow", "Carve"))

