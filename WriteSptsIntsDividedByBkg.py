"""This function removes the background from spots intensity.

Given a folder with analyzed data, the algorithm evaluates the
background value of each spot in each frame (using cages) and
subtracte it from the intensity of the spot. The background value
is multiplied by the volume of the spots of course. results
written in a .xls file.
"""

import datetime
import multiprocessing
import numpy as np
from skimage.measure import regionprops, regionprops_table
import xlsxwriter


class WriteSptsIntsDividedByBkg:
    """Main class that calculates the background value for each nucleus and writes results into excel file"""
    def __init__(self, foldername, green4D, spots_3D, spots_tracked_3D):

        spts_id  =  spots_tracked_3D[spots_tracked_3D != 0]
        spts_id  =  np.unique(spts_id).astype(np.uint16)
        steps    =  spots_tracked_3D.shape[0]

        cpu_ow     =  multiprocessing.cpu_count()
        jobs_args  =  []
        t_chops    =  np.linspace(0, steps, cpu_ow).astype(int)
        for k in range(cpu_ow - 1):
            jobs_args.append([green4D[t_chops[k]:t_chops[k + 1], :, :, :], spots_3D.spots_ints[t_chops[k]:t_chops[k + 1], :, :], spots_3D.spots_vol[t_chops[k]:t_chops[k + 1], :, :], spots_tracked_3D[t_chops[k]:t_chops[k + 1], :, :], spts_id])

        pool     =  multiprocessing.Pool()
        results  =  pool.map(WriteSptsIntsDividedByBkgUtility, jobs_args)
        pool.close()

        spts_int_bybkg     =  results[0].spts_int_bybkg
        av_spts_int_bybkg  =  results[0].av_spts_int_bybkg
        bkg                =  results[0].bkg
        bkg_variab         =  results[0].bkg_variab
        for j in range(1, cpu_ow - 1):
            spts_int_bybkg     =  np.concatenate((spts_int_bybkg, results[j].spts_int_bybkg), axis=1)
            bkg                =  np.concatenate((bkg, results[j].bkg), axis=1)
            bkg_variab         =  np.concatenate((bkg_variab, results[j].bkg_variab), axis=1)
            av_spts_int_bybkg  =  np.concatenate((av_spts_int_bybkg, results[j].av_spts_int_bybkg), axis=1)

        del results

        spts_int_bybkg[np.isnan(spts_int_bybkg)]        =  0
        av_spts_int_bybkg[np.isnan(av_spts_int_bybkg)]  =  0

        self.bkg             =  bkg
        self.bkg_variab      =  bkg_variab
        self.spts_int_bybkg  =  spts_int_bybkg

        book    =  xlsxwriter.Workbook(foldername + '/SpotsIntensityDividedByBackground.xlsx')
        sheet1  =  book.add_worksheet("Spots by Background")
        sheet2  =  book.add_worksheet("Average Background")
        sheet3  =  book.add_worksheet("Background Std")
        sheet4  =  book.add_worksheet("Average Spots by Background")

        sheet1.write(0, 0, "Folder Name")
        sheet1.write(0, 1, foldername)
        sheet1.write(1, 0, "date")
        sheet1.write(1, 1, datetime.datetime.now().strftime("%d-%b-%Y"))

        for k in range(spts_id.size):
            sheet1.write(0, 3 + k, "SptId_" + str(spts_id[k]))
            sheet2.write(0, 3 + k, "SptId_" + str(spts_id[k]))
            sheet3.write(0, 3 + k, "SptId_" + str(spts_id[k]))
            sheet4.write(0, 3 + k, "SptId_" + str(spts_id[k]))
            for t in range(steps):
                sheet1.write(t + 1, 3 + k, spts_int_bybkg[k, t])
                try:
                    sheet2.write(t + 1, 3 + k, bkg[k, t, 0])
                except TypeError:
                    print(t)
                try:
                    sheet3.write(t + 1, 3 + k, bkg_variab[k, t])
                except TypeError:
                    pass
                sheet4.write(t + 1, 3 + k, av_spts_int_bybkg[k, t])

        book.close()


class WriteSptsIntsDividedByBkgUtility:
    """Calculate background cells in smalle regions, for multiprocessing pourposes"""
    def __init__(self, input_args):                      # list is made by raw data, ints data, vol data, tag data, tag list

        steps, z_tot, x_tot, y_tot  =  input_args[0].shape
        spts_int_bybkg              =  np.zeros((input_args[4].size, steps))
        av_spts_int_bybkg           =  np.zeros((input_args[4].size, steps))
        bkg                         =  np.zeros((input_args[4].size, steps, 2))
        bkg_variab                  =  np.zeros((input_args[4].size, steps))

        for t in range(steps):
            rgp_spts      =  regionprops(input_args[3][t].astype(np.uint16))
            bkg_tags      =  np.zeros(input_args[0][t].shape, dtype=np.uint16)
            for j in range(len(rgp_spts)):
                x_ctr, y_ctr  =  rgp_spts[j]['centroid']
                x_ctr, y_ctr  =  int(round(x_ctr)), int(round(y_ctr))
                z_ctr         =  np.argmax(input_args[0][t, :, x_ctr, y_ctr])                        # z center coordinate

                z_min_ext  =  np.max([z_ctr - 5, 0])                                           # edges of the cage, internal and external. Controls for spots close to the borders
                z_max_ext  =  np.min([z_ctr + 6, z_tot])                                       # in zed the edge is smaller beacause in z the step is smaller than in x or y
                z_min_int  =  np.max([z_ctr - 3, 0])
                z_max_int  =  np.min([z_ctr + 4, z_tot])

                x_min_ext  =  np.max([x_ctr - 9, 0])
                x_max_ext  =  np.min([x_ctr + 10, x_tot])
                x_min_int  =  np.max([x_ctr - 7, 0])
                x_max_int  =  np.min([x_ctr + 8, x_tot])

                y_min_ext  =  np.max([y_ctr - 9, 0])
                y_max_ext  =  np.min([y_ctr + 10, y_tot])
                y_min_int  =  np.max([y_ctr - 7, 0])
                y_max_int  =  np.min([y_ctr + 8, y_tot])

                bkg_tags[z_min_ext:z_max_ext, x_min_ext:x_max_ext, y_min_ext:y_max_ext]  =  rgp_spts[j]["label"]                # cage definition (contains more than 1400 pixels)
                bkg_tags[z_min_int:z_max_int, x_min_int:x_max_int, y_min_int:y_max_int]  =  0
                bkg[np.where(input_args[4] == rgp_spts[j]["label"])[0][0], t, 1]         =  z_ctr

            try:
                rgp_cages  =  regionprops_table(bkg_tags, input_args[0][t], properties=["label", "intensity_image"])
                for count, lbl in enumerate(rgp_cages["label"]):
                    cages_ints                     =  rgp_cages["intensity_image"][count]
                    cages_ints                     =  cages_ints[cages_ints != 0]
                    idx_tag                        =  np.where(input_args[4] == lbl)[0][0]
                    spts_int_bybkg[idx_tag, t]     =  np.sum(input_args[1][t] * (input_args[3][t] == lbl)) / np.mean(cages_ints)
                    av_spts_int_bybkg[idx_tag, t]  =  spts_int_bybkg[idx_tag, t] / np.sum(input_args[2][t] * (input_args[3][t] == lbl))
                    bkg[idx_tag, t, 0]             =  cages_ints.mean()
                    bkg_variab[idx_tag, t]         =  cages_ints.std()

            except IndexError:                                                                                                      # in case of empty frames
                pass

        self.spts_int_bybkg     =  spts_int_bybkg
        self.bkg                =  bkg
        self.bkg_variab         =  bkg_variab
        self.av_spts_int_bybkg  =  av_spts_int_bybkg
