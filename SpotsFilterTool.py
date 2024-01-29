"""Stand-alone post-processing tool to treat the results of 'SegmentTrack_v4.0'

https://github.com/ant-trullo/SegmentTrack_v4.0

author antonio.trullo@igmm.cnrs.fr

In this post-processing tool the user can set binary filter with different
parameters in differen parts of the movie.
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from skimage.morphology import label

import AnalysisLoader
import SpotsSeveralFilters
import AnalysisSaver


class SpotsFilterTool(QtWidgets.QWidget):
    """Pop up tool to save the tags of the sibiling to remove."""
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        ksf_h, ksf_w  =  1, 1

        analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))

        raw_data     =  AnalysisLoader.RawData(analysis_folder)
        spts_track   =  np.load(analysis_folder + '/spots_tracked.npy')
        nucs_track   =  np.load(analysis_folder + '/nuclei_tracked.npy') * (1 - np.sign(spts_track))
        spts_rmvd    =  np.zeros_like(spts_track)
        spots2show   =  np.sign(spts_track) + np.sign(spts_rmvd) + np.sign(nucs_track) * 3
        bin_cmap     =  np.zeros((4, 3), dtype='uint16')
        bin_cmap[0]  =  [0, 0, 0]
        bin_cmap[1]  =  [0, 255, 0]
        bin_cmap[2]  =  [255, 0, 0]
        bin_cmap[3]  =  [0, 0, 255]
        mycmap       =  pg.ColorMap(np.linspace(0, 1, 4), color=bin_cmap)

        frame1  =  pg.ImageView(self, name="Frame1")
        frame1.ui.menuBtn.hide()
        frame1.ui.roiBtn.hide()
        frame1.setImage(raw_data.imarray_green, autoRange=True)
        frame1.timeLine.sigPositionChanged.connect(self.update_from_frame1)

        frame2  =  pg.ImageView(self, name="Frame2")
        frame2.ui.menuBtn.hide()
        frame2.ui.roiBtn.hide()
        frame2.ui.histogram.hide()
        frame2.setImage(spots2show, autoRange=True)
        frame2.view.setXLink("Frame1")
        frame2.view.setYLink("Frame1")
        frame2.timeLine.sigPositionChanged.connect(self.update_from_frame2)
        frame2.setColorMap(mycmap)
        frame2.getImageItem().mouseClickEvent  =  self.click

        one_spot_checkbox  =  QtWidgets.QCheckBox(self)
        one_spot_checkbox.setFixedSize(int(ksf_h * 120), int(ksf_h * 25))
        one_spot_checkbox.setText("1 Spot")
        one_spot_checkbox.setToolTip("Keeps only the biggest spot per nucleus")
        one_spot_checkbox.stateChanged.connect(self.one_spot_checkbox_changes)

        numb_zeros_lbl  =  QtWidgets.QLabel("# 0", self)
        numb_zeros_lbl.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        first_time_lbl  =  QtWidgets.QLabel("Start", self)
        first_time_lbl.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        last_time_lbl  =  QtWidgets.QLabel("End", self)
        last_time_lbl.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        numb_zeros_first_edt  =  QtWidgets.QLineEdit(self)
        numb_zeros_first_edt.textChanged[str].connect(self.numb_zeros_frst_var)
        numb_zeros_first_edt.setToolTip("Number of inactive frames before and after an active one in the first time slot to identify a spot to remove")
        numb_zeros_first_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        slot_strt_frst_edt = QtWidgets.QLineEdit(self)
        slot_strt_frst_edt.textChanged[str].connect(self.slot_strt_frst_var)
        slot_strt_frst_edt.setText("0")
        slot_strt_frst_edt.setToolTip("Starting frame of the first time slot")
        slot_strt_frst_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))
        slot_strt_frst_edt.setEnabled(False)

        slot_end_frst_edt  =  QtWidgets.QLineEdit(self)
        slot_end_frst_edt.textChanged[str].connect(self.slot_end_frst_var)
        slot_end_frst_edt.setToolTip("Last frame of the first time slot")
        slot_end_frst_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        numb_zeros_scnd_edt  =  QtWidgets.QLineEdit(self)
        numb_zeros_scnd_edt.textChanged[str].connect(self.numb_zeros_scnd_var)
        numb_zeros_scnd_edt.setToolTip("Number of inactive frames before and after an active one in the second time slot to identify a spot to remove")
        numb_zeros_scnd_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        slot_strt_scnd_edt = QtWidgets.QLineEdit(self)
        slot_strt_scnd_edt.textChanged[str].connect(self.slot_strt_scnd_var)
        slot_strt_scnd_edt.setToolTip("Starting frame of the second time slot")
        slot_strt_scnd_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        slot_end_scnd_edt = QtWidgets.QLineEdit(self)
        slot_end_scnd_edt.textChanged[str].connect(self.slot_end_scnd_var)
        slot_end_scnd_edt.setToolTip("Last frame of the second time slot")
        slot_end_scnd_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        numb_zeros_thrd_edt  =  QtWidgets.QLineEdit(self)
        numb_zeros_thrd_edt.textChanged[str].connect(self.numb_zeros_thrd_var)
        numb_zeros_thrd_edt.setToolTip("Number of inactive frames before and after an active one in the third time slot to identify a spot to remove")
        numb_zeros_thrd_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        slot_strt_thrd_edt  =  QtWidgets.QLineEdit(self)
        slot_strt_thrd_edt.textChanged[str].connect(self.slot_strt_thrd_var)
        slot_strt_thrd_edt.setToolTip("Starting frame of the third time slot")
        slot_strt_thrd_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))

        slot_end_thrd_edt  =  QtWidgets.QLineEdit(self)
        slot_end_thrd_edt.textChanged[str].connect(self.slot_end_thrd_var)
        slot_end_thrd_edt.setToolTip("Last frame of the third time slot")
        slot_end_thrd_edt.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))
        slot_end_thrd_edt.setText(str(raw_data.imarray_green.shape[0]))
        slot_end_thrd_edt.setEnabled(False)

        filt_settings_grid  =  QtWidgets.QGridLayout()
        filt_settings_grid.addWidget(numb_zeros_lbl, 0, 0)
        filt_settings_grid.addWidget(first_time_lbl, 0, 1)
        filt_settings_grid.addWidget(last_time_lbl, 0, 2)
        filt_settings_grid.addWidget(numb_zeros_first_edt, 1, 0)
        filt_settings_grid.addWidget(slot_strt_frst_edt, 1, 1)
        filt_settings_grid.addWidget(slot_end_frst_edt, 1, 2)
        filt_settings_grid.addWidget(numb_zeros_scnd_edt, 2, 0)
        filt_settings_grid.addWidget(slot_strt_scnd_edt, 2, 1)
        filt_settings_grid.addWidget(slot_end_scnd_edt, 2, 2)
        filt_settings_grid.addWidget(numb_zeros_thrd_edt, 3, 0)
        filt_settings_grid.addWidget(slot_strt_thrd_edt, 3, 1)
        filt_settings_grid.addWidget(slot_end_thrd_edt, 3, 2)

        solidity_thr_lbl  =  QtWidgets.QLabel("Sol Thr", self)
        solidity_thr_lbl.setFixedSize(int(ksf_h * 75), int(ksf_w * 25))

        solidity_thr_edt  =  QtWidgets.QLineEdit(self)
        solidity_thr_edt.textChanged[str].connect(self.solidity_thr_var)
        solidity_thr_edt.setToolTip("Set the solidity threshold; suggested value ~4")
        solidity_thr_edt.setFixedSize(int(ksf_h * 25), int(ksf_w * 25))
        solidity_thr_edt.setEnabled(False)

        solidity_thr_box  =  QtWidgets.QHBoxLayout()
        solidity_thr_box.addWidget(solidity_thr_lbl)
        solidity_thr_box.addWidget(solidity_thr_edt)

        spots_filter_btn  =  QtWidgets.QPushButton("Filter")
        spots_filter_btn.setToolTip("Filter spots with the parameters you set")
        spots_filter_btn.setFixedSize(int(ksf_h * 120), int(ksf_w * 25))
        spots_filter_btn.clicked[bool].connect(self.spots_filter)

        save_filtspts_btn  =  QtWidgets.QPushButton("Save", self)
        save_filtspts_btn.clicked.connect(self.save_filtspts)
        save_filtspts_btn.setToolTip("Save filtered time series")
        save_filtspts_btn.setFixedSize(int(ksf_h * 120), int(ksf_w * 25))

        frame_numb_lbl  =  QtWidgets.QLabel('Frame 0', self)
        frame_numb_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        groupbox  =  QtWidgets.QGroupBox("Filter Settings")
        groupbox.setCheckable(False)
        groupbox.setFixedSize(int(ksf_h * 240), int(ksf_w * 230))
        groupbox.setLayout(filt_settings_grid)

        commands  =  QtWidgets.QVBoxLayout()
        commands.addWidget(one_spot_checkbox)
        commands.addLayout(solidity_thr_box)
        commands.addWidget(groupbox)
        commands.addWidget(spots_filter_btn)
        commands.addStretch()
        commands.addWidget(save_filtspts_btn)
        commands.addWidget(frame_numb_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addWidget(frame1)
        layout.addWidget(frame2)
        layout.addLayout(commands)

        self.frame1              =  frame1
        self.frame2              =  frame2
        self.one_spot_checkbox   =  one_spot_checkbox
        self.nucs_track          =  nucs_track
        self.spts_track          =  spts_track
        self.spots2show          =  spots2show
        self.frame_numb_lbl      =  frame_numb_lbl
        self.spots_filter_btn    =  spots_filter_btn
        self.save_filtspts_btn   =  save_filtspts_btn
        self.analysis_folder     =  analysis_folder
        self.raw_data            =  raw_data
        self.solidity_thr_edt    =  solidity_thr_edt
        self.spts2rm             =  np.zeros_like(spts_track)
        self.slot_end_frst_edt   =  slot_end_frst_edt
        self.slot_strt_scnd_edt  =  slot_strt_scnd_edt
        self.slot_end_scnd_edt   =  slot_end_scnd_edt
        self.slot_strt_thrd_edt  =  slot_strt_thrd_edt

        self.setLayout(layout)
        self.setGeometry(30, 30, 800, 100)
        self.setWindowTitle('Spots Filter')
        self.show()

    def update_from_frame1(self):
        """Synchronize the frame2 from frame1."""
        self.frame2.setCurrentIndex(self.frame1.currentIndex)
        self.frame_numb_lbl.setText("Frame " + str(self.frame1.currentIndex))

    def update_from_frame2(self):
        """Synchronize the frame1 from frame2."""
        self.frame1.setCurrentIndex(self.frame2.currentIndex)

    def numb_zeros_frst_var(self, text):
        """Set the number of inactive frames around an active one in the first chop."""
        self.numb_zeros_frst_value  =  int(text)

    def numb_zeros_scnd_var(self, text):
        """Set the number of inactive frames around an active one in the second chop."""
        self.numb_zeros_scnd_value  =  int(text)

    def numb_zeros_thrd_var(self, text):
        """Set the number of inactive frames around an active one in the third chop."""
        self.numb_zeros_thrd_value  =  int(text)

    def slot_strt_frst_var(self, text):
        """Set the start frame for the first filter slot."""
        self.slot_strt_frst_value  =  int(text)

    def slot_end_frst_var(self, text):
        """Set the end frame for the first filter slot and the first of the second slot."""
        self.slot_end_frst_value  =  int(text)
        self.slot_strt_scnd_edt.setText(str(int(text) + 1))

    def slot_strt_scnd_var(self, text):
        """Set the start frame for the second filter slot and the end frame of the first filter slot."""
        self.slot_strt_scnd_value  =  int(text)
        self.slot_end_frst_edt.setText(str(int(text) - 1))

    def slot_end_scnd_var(self, text):
        """Set the end frame for the second filter slot and the start frame of the third filter slot."""
        self.slot_end_scnd_value  =  int(text)
        self.slot_strt_thrd_edt.setText(str(int(text) + 1))

    def slot_strt_thrd_var(self, text):
        """Set the start frame for the third filter slot and the end frame for the second filter slot."""
        self.slot_strt_thrd_value  =  int(text)
        self.slot_end_scnd_edt.setText(str(int(text) - 1))

    def slot_end_thrd_var(self, text):
        """Set the end frame for the third filter slot."""
        self.slot_end_thrd_value  =  int(text)

    def solidity_thr_var(self, text):
        """Set the solidity threshold."""
        self.solidity_thr_value  =  int(text)

    def one_spot_checkbox_changes(self):
        """Change solidity enable state according to the checkbox choice."""
        if self.one_spot_checkbox.checkState() != 0:
            self.solidity_thr_edt.setEnabled(True)
        else:
            self.solidity_thr_edt.setEnabled(False)

    def spots_filter(self):
        """Filter spots."""
        self.spots_filter_btn.setStyleSheet("background-color : red")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        if self.one_spot_checkbox.checkState() != 0:
            spts2rm_1        =  SpotsSeveralFilters.OneSpotsPerNucleus(self.spts_track, self.solidity_thr_value).spts2rm
            spts2rm_2        =  SpotsSeveralFilters.RemoveIsolatedSpots(self.spts_track * (1 - spts2rm_1), self.numb_zeros_frst_value, self.numb_zeros_scnd_value, self.numb_zeros_thrd_value, self.slot_strt_frst_value, self.slot_end_frst_value, self.slot_strt_scnd_value, self.slot_end_scnd_value, self.slot_strt_thrd_value, self.slot_end_thrd_value).spts2rm
            self.spts2rm     =  np.sign(spts2rm_1 + spts2rm_2)
            self.spots2show  =  np.sign(self.spts_track)  +  self.spts2rm  +  np.sign(self.nucs_track) * 3
        else:
            self.spts2rm     =  SpotsSeveralFilters.RemoveIsolatedSpots(self.spts_track, self.numb_zeros_frst_value, self.numb_zeros_scnd_value, self.numb_zeros_thrd_value, self.slot_strt_frst_value, self.slot_end_frst_value, self.slot_strt_scnd_value, self.slot_end_scnd_value, self.slot_strt_thrd_value, self.slot_end_thrd_value).spts2rm
            self.spots2show  =  np.sign(self.spts_track)  +  self.spts2rm  +  np.sign(self.nucs_track) * 3

        cif  =  self.frame1.currentIndex
        self.frame2.setImage(self.spots2show, autoRange=False)
        self.frame2.setCurrentIndex(cif)
        self.spots_filter_btn.setStyleSheet("background-color : white")

    def save_filtspts(self):
        """Save the results of the filters."""
        self.save_filtspts_btn.setStyleSheet("background-color : red")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        spots_3D          =  SpotsSeveralFilters.FilteredSpots2Save(self.analysis_folder, self.spts2rm)
        AnalysisSaver.FilteredSpotsSaver(self.analysis_folder, spots_3D, self.raw_data.fnames, self.raw_data.green4D, self.solidity_thr_value, self.numb_zeros_frst_value, self.numb_zeros_scnd_value, self.numb_zeros_thrd_value, self.slot_strt_frst_value, self.slot_end_frst_value, self.slot_strt_scnd_value, self.slot_end_scnd_value, self.slot_strt_thrd_value, self.slot_end_thrd_value)
        self.save_filtspts_btn.setStyleSheet("background-color : white")

    def click(self, event):
        """Select a spot to change its selection."""
        event.accept()
        pos        =  [int(event.pos()[0]), int(event.pos()[1])]
        modifiers  =  QtWidgets.QApplication.keyboardModifiers()

        if modifiers  ==  QtCore.Qt.ShiftModifier:
            spts_bff  =  label(self.spts_track[self.frame2.currentIndex])
            if self.spts2rm[self.frame2.currentIndex, pos[0], pos[1]] == 0:
                self.spts2rm[self.frame2.currentIndex]     +=  (spts_bff == spts_bff[pos[0], pos[1]])
                self.spots2show[self.frame2.currentIndex]  +=  (spts_bff == spts_bff[pos[0], pos[1]])
            elif self.spts2rm[self.frame2.currentIndex, pos[0], pos[1]] == 1:
                self.spts2rm[self.frame2.currentIndex]     -=  (spts_bff == spts_bff[pos[0], pos[1]])
                self.spots2show[self.frame2.currentIndex]  -=  (spts_bff == spts_bff[pos[0], pos[1]])
            self.frame2.updateImage()

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":

    sys.excepthook  =  except_hook

    app     =  QtWidgets.QApplication(sys.argv)
    window  =  SpotsFilterTool()
    window.show()
    sys.exit(app.exec_())
