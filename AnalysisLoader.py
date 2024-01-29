"""This function loads the previously done analysis."""


import os
import numpy as np
from PyQt5 import QtWidgets

import MultiLoadCzi5D


class RawData:
    """Load raw data file."""
    def __init__(self, foldername):

        fnames       =  QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", foldername, filter="*.lsm *.czi *.tif *.lif")[0]
        raw_data     =  MultiLoadCzi5D.MultiProcLoadCzi5D(fnames, np.fromfile(foldername + '/nucs_spts_ch.bin', 'uint16'))
        if os.path.isfile(foldername + '/crop_vect.npy'):
            crop_vect  =  np.load(foldername + '/crop_vect.npy')
        else:
            crop_vect  =  np.array([0, 0, raw_data.imarray_red.shape[1], raw_data.imarray_red.shape[2]])

        raw_data.imarray_green  =  raw_data.imarray_green[:, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]
        raw_data.imarray_red    =  raw_data.imarray_red[:, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]
        raw_data.green4D        =  raw_data.green4D[:, :, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]

        im_red_smpl  =  np.load(foldername + '/im_red_smpl.npy')
        jj_start     =  np.where(np.sum(raw_data.imarray_red - im_red_smpl[0], axis=(1, 2)) == 0)[0][0]
        jj_end       =  np.where(np.sum(raw_data.imarray_red - im_red_smpl[1], axis=(1, 2)) == 0)[0][0]

        self.imarray_green  =  raw_data.imarray_green[jj_start:jj_end + 1]
        self.imarray_red    =  raw_data.imarray_red[jj_start:jj_end + 1]
        self.green4D        =  raw_data.green4D[jj_start:jj_end + 1]
        self.pix_size       =  raw_data.pix_size
        self.pix_size_Z     =  raw_data.pix_size_Z
        self.fnames         =  fnames
        self.crop_vect      =  crop_vect
