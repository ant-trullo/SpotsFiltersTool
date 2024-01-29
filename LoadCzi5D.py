"""This function loads .lsm files.

Given the filename of a .lsm file, this function gives as output the matrices
of the red and green channels maximum intensity projected plus the green channel
as it is. Inputs are the file-name and the channel number for nuclei and spots.
"""


import numpy as np
import czifile


class LoadCzi5D:
    """Only class, does all the job."""
    def __init__(self, fname, nucs_spts_ch):

        file_array  =  np.squeeze(czifile.imread(fname))

        if len(file_array.shape)  == 5:                                                             # case you have more than a time frame
            c, steps, z, x_len, y_len  =  file_array.shape

            red_mtx    =  np.zeros((steps, x_len, y_len), dtype=np.int32)
            green_mtx  =  np.zeros((steps, x_len, y_len), dtype=np.int32)

            for t in range(steps):                                                                  # maximum intensity projection
                for x in range(x_len):
                    red_mtx[t, x, :]    =  file_array[nucs_spts_ch[0], t, :, x, :].max(0)
                    green_mtx[t, x, :]  =  file_array[nucs_spts_ch[1], t, :, x, :].max(0)
            self.green4D  =  file_array[nucs_spts_ch[1], :, :, :, :]

        else:                                                                                       # case you have just one time frame
            c, z, x_len, y_len  =  file_array.shape

            red_mtx    =  np.zeros((x_len, y_len))
            green_mtx  =  np.zeros((x_len, y_len))

            for x in range(x_len):                                                                  # maximum intensity projection
                red_mtx[x, :]    =  file_array[nucs_spts_ch[0], :, x, :].max(0)
                green_mtx[x, :]  =  file_array[nucs_spts_ch[1], :, x, :].max(0)

            self.green4D    =  file_array[nucs_spts_ch[1], :, :, :]

        self.red_mtx    =  red_mtx
        self.green_mtx  =  green_mtx
