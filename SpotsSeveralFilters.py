"""This function filters tracked spots.

Input is the tracked spots and filter settings.
"""


import multiprocessing
import numpy as np
from skimage.measure import regionprops_table, label
from skimage.morphology import convex_hull_image
from PyQt5 import QtWidgets

# import ServiceWidgets


def build3d_from_coords(spots_coords, t):
    """Build 3d (zxy) spots from coords."""
    spots_coords_t  =  spots_coords[spots_coords[:, 0] == t]                                                            # isolate coordinates with a determined t (t is input)
    spts_fr         =  np.zeros((spots_coords[-1, 1], spots_coords[-1, 2], spots_coords[-1, 3]), dtype=np.uint16)       # initialize the output matrix: the size is in the last raw of the coordinates matrix
    for kk in spots_coords_t[:, 1:]:                                                                                    # fill the initialized matrix puting 1 in the position of the selected coordinates
        spts_fr[kk[0], kk[1], kk[2]]  =  1
    return spts_fr


class OneSpotsPerNucleus:
    """Keeps only one spot per nucleus, the biggest one."""
    def __init__(self, spts_track, solidity_thr):

        tlen        =  spts_track.shape[0]                                                                              # number of time frames
        cpu_owe     =  min(multiprocessing.cpu_count(), 16)                                                             # number of cpu (maximum of 16 for Virginia pc)
        jobs_args   =  []                                                                                               # initialize the list of the arguments
        sub_frames  =  np.array_split(np.arange(tlen), cpu_owe)                                                         # split the total number of time frames in nearly egual chops
        for sub in sub_frames:
            jobs_args.append([spts_track[sub], solidity_thr])                                                           # populate the list

        pool     =  multiprocessing.Pool()                                                                              # launch the multiprocessing pool
        results  =  pool.map(OneSpotsPerNucleusUtility, jobs_args)

        spts2rm  =  np.zeros_like(spts_track)                                                                           # initialize the output matrix
        for cnt, sub_frame in enumerate(sub_frames):
            spts2rm[sub_frame]  =  results[cnt].spts2rm

        self.spts2rm  =  spts2rm


class OneSpotsPerNucleusUtility:
    """Keeps only one spot per nucleus, the biggest one."""
    def __init__(self, job_args):

        spts_track    =  job_args[0]
        solidity_thr  =  job_args[1]

        tlen     =  spts_track.shape[0]                                                                                 # number of time steps
        spts2rm  =  np.zeros_like(spts_track)                                                                           # initialize the matrix of the spots to remove

        for tt in range(tlen):
            rgp       =  regionprops_table(spts_track[tt], properties=["label", "area"])                                # regionprops of the tracked spots
            cv_im     =  []
            for mm in range(len(rgp["label"])):
                cv_im.append(np.sum(convex_hull_image((spts_track[tt] == rgp["label"][mm]).astype(np.uint8))))          # for each label determine the convex hull image area
            solidity  =  np.asarray(cv_im) / rgp["area"]                                                                # two spots far from each other have a convew hull image bigger than the sum of the bare spots
            idxs2wrk  =  np.where(solidity > solidity_thr)[0]                                                           # for a solid spot sld = 1; when sld higher than one, the spots is split in several connected components.
                                                                                                                        # Tuning the threshold is possible to differentiate between sister chromatids and detection errors.
                                                                                                                        # Even for a solid single spot you can have this ratio slightly higher: happens since spots are not regular
            for idx2wrk in idxs2wrk:                                                                                    # for each of the high sld spots
                spt      =  (spts_track[tt] == rgp["label"][idx2wrk]) * 1                                               # isolate it
                spt_lbl  =  label(spt)                                                                                  # label it
                if spt_lbl.max() > 1:                                                                                   # if there is more than 1 label
                    rgp_spt   =  regionprops_table(spt_lbl, properties=["label", "area"])                               # regionprop to measure area
                    idxs_bff  =  list(np.arange(spt_lbl.max()))                                                         # list of the indexes of the identified connected components
                    idxs_bff.remove(np.argmax(rgp_spt["area"]))                                                         # remove from the list the index of the connectedc component with bigger surface
                    for idx_bff in idxs_bff:
                        spts2rm[tt]  +=  spt_lbl == rgp_spt["label"][idx_bff]                                           # add to the final matrix the spots to be removed

        self.spts2rm  =  spts2rm


class FilteredSpots2Save:
    """Organize filtered spots in a class with attributes as in the Chopper Spots detector."""
    def __init__(self, analysis_folder, spts2rm):

        self.spots_ints   =  np.load(analysis_folder + '/spots_3D_ints.npy') * (1 - spts2rm)                            # load ints matrices and remove the spots to remove
        self.spots_vol    =  np.load(analysis_folder + '/spots_3D_vol.npy') * (1 - spts2rm)                             # load vol matrices and remove the spots to remove
        spots_coords_old  =  np.load(analysis_folder + '/spots_3D_coords.npy')                                          # load spots_coords old matrix (it will be corrected)

        spots_coords  =  np.zeros((0, 4), dtype=np.int16)                                                               # initialize spots_coords matrix
        spots_tzxy    =  np.zeros((0, 4), dtype=np.int16)                                                               # initialize spots_tzxy matrix

        tlen, zlen, xlen, ylen  =  spots_coords_old[-1]                                                                 # 4D shape of the data
        for tt in range(tlen):
            spts_bff_tt   =  build3d_from_coords(spots_coords_old, tt)
            spts_bff_tt  *=  (1 - spts2rm[tt])
            spts_bff_tt   =  label(spts_bff_tt)
            rgp_bff       =  regionprops_table(spts_bff_tt, properties=["centroid", "coords"])
            for k in range(len(rgp_bff["centroid-0"])):
                spots_tzxy    =  np.concatenate([spots_tzxy, [np.append(tt, np.round(np.array([rgp_bff['centroid-0'][k], rgp_bff['centroid-1'][k], rgp_bff['centroid-2'][k]])).astype(np.uint16))]], axis=0)  # coordinates of the good spots
                spots_coords  =  np.concatenate((spots_coords, np.column_stack((tt * np.ones(rgp_bff['coords'][k].shape[0], dtype=np.int16), rgp_bff['coords'][k]))), axis=0)

        self.spots_coords  =  np.concatenate([spots_coords, np.array([[tlen, zlen, xlen, ylen]])])
        self.spots_tzxy    =  spots_tzxy


class RemoveIsolatedSpots:
    """Remove spots appearing only from an isolated frames."""
    def __init__(self, spts_track, numb_zeros_frst_value, numb_zeros_scnd_value, numb_zeros_thrd_value, slot_strt_frst_value, slot_end_frst_value, slot_strt_scnd_value, slot_end_scnd_value, slot_strt_thrd_value, slot_end_thrd_value):

        spts2rm    =  np.zeros_like(spts_track)                                                                         # initialize spots to remove matrix
        spts_idxs  =  np.unique(spts_track[spts_track != 0])                                                            # indexes of the tracked spots
        pbar       =  ProgressBar(total1=spts_idxs.size)
        pbar.show()

        for cnt, spt_idx in enumerate(spts_idxs):                                                                       # for each spot
            pbar.update_progressbar(cnt)
            prof  =  np.sign(np.sum(spts_track == spt_idx, axis=(1, 2)))                                                # 0/1 profile (active-inactive)

            prof1       =  np.concatenate([np.zeros(numb_zeros_frst_value), prof[slot_strt_frst_value:slot_end_frst_value]])     # padd numb_zeros_frst zeros at the beginning to deal with isolated spot at the beginning
            test_list1  =  list(np.concatenate([np.zeros(numb_zeros_frst_value, dtype=np.int64), np.array([1]), np.zeros(numb_zeros_frst_value, dtype=np.int64)]))  # pattern of active inactive to search for in the profile
            t2rm1       =  [i for i in range(0, len(prof1)) if list(prof1[i:i + 1 + 2 * numb_zeros_frst_value]) == test_list1]              # search for the pattern in the profile time series
            for tt in t2rm1:
                spts2rm[tt]  +=  (spts_track[tt] == spt_idx)                                  # remove the spot in the isolated active frame

            prof2       =  prof[slot_strt_scnd_value - numb_zeros_scnd_value:slot_end_scnd_value]           # take num_of_zeros frames before too to avoid the filter to not work properly across the region changes
            test_list2  =  list(np.concatenate([np.zeros(numb_zeros_scnd_value, dtype=np.int64), np.array([1]), np.zeros(numb_zeros_scnd_value, dtype=np.int64)]))  # pattern of active inactive to search for in the profile
            t2rm2       =  [i for i in range(0, len(prof2)) if list(prof2[i:i + 1 + 2 * numb_zeros_scnd_value]) == test_list2]              # search for the pattern in the profile time series
            t2rm2       =  np.asarray(t2rm2) + slot_strt_scnd_value
            for tt in t2rm2:
                spts2rm[tt]  +=  (spts_track[tt] == spt_idx)                                  # remove the spot in the isolated active frame

            prof3       =  np.concatenate([prof[slot_strt_thrd_value - numb_zeros_thrd_value:slot_end_thrd_value], np.zeros(numb_zeros_thrd_value)])    # take numb_of_zeros frames before to avoid the filter to not work across region changes and pass numb_of_zeros after to delete isolated last activations
            test_list3  =  list(np.concatenate([np.zeros(numb_zeros_thrd_value, dtype=np.int64), np.array([1]), np.zeros(numb_zeros_thrd_value, dtype=np.int64)]))  # pattern of active inactive to search for in the profile
            t2rm3       =  [i for i in range(0, len(prof1)) if list(prof3[i:i + 1 + 2 * numb_zeros_thrd_value]) == test_list3]              # search for the pattern in the profile time series
            t2rm3       =  np.asarray(t2rm3) + slot_strt_thrd_value
            for tt in t2rm3:
                spts2rm[tt]  +=  (spts_track[tt] == spt_idx)                                  # remove the spot in the isolated active frame

        pbar.close()
        self.spts2rm  =  spts2rm


class ProgressBar(QtWidgets.QWidget):
    """Simple progressbar widget."""
    def __init__(self, parent=None, total1=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar(self, val1):
        """Update progressbar."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()


        # for cnt, spt_idx in enumerate(spts_idxs):                                                                       # for each spot
        #     pbar.update_progressbar1(cnt)
        #     prof  =  np.sign(np.sum(spts_track == spt_idx, axis=(1, 2)))                                                # 0/1 profile (active-inactive)
        #
        #     prof1       =  prof[slot_strt_frst_value:slot_end_frst_value]
        #     test_list1  =  list(np.concatenate([np.zeros(numb_zeros_frst_value, dtype=np.int64), np.array([1]), np.zeros(numb_zeros_frst_value, dtype=np.int64)]))  # pattern of active inactive to search for in the profile
        #     t2rm1       =  [i for i in range(0, len(prof1)) if list(prof1[i:i + 1 + 2 * numb_zeros_frst_value]) == test_list1]              # search for the pattern in the profile time series
        #     for tt in t2rm1:
        #         spts2rm[tt + numb_zeros_frst_value]  +=  (spts_track[tt + numb_zeros_frst_value] == spt_idx)                                  # remove the spot in the isolated active frame
        #
        #     prof2       =  prof[slot_strt_scnd_value:slot_end_scnd_value]
        #     test_list2  =  list(np.concatenate([np.zeros(numb_zeros_scnd_value, dtype=np.int64), np.array([1]), np.zeros(numb_zeros_scnd_value, dtype=np.int64)]))  # pattern of active inactive to search for in the profile
        #     t2rm2       =  [i for i in range(0, len(prof2)) if list(prof2[i:i + 1 + 2 * numb_zeros_scnd_value]) == test_list2]              # search for the pattern in the profile time series
        #     t2rm2       =  np.asarray(t2rm2) + slot_strt_scnd_value
        #     for tt in t2rm2:
        #         spts2rm[tt + numb_zeros_scnd_value]  +=  (spts_track[tt + numb_zeros_scnd_value] == spt_idx)                                  # remove the spot in the isolated active frame
        #
        #     prof3       =  prof[slot_strt_thrd_value:slot_end_thrd_value]
        #     test_list3  =  list(np.concatenate([np.zeros(numb_zeros_thrd_value, dtype=np.int64), np.array([1]), np.zeros(numb_zeros_thrd_value, dtype=np.int64)]))  # pattern of active inactive to search for in the profile
        #     t2rm3       =  [i for i in range(0, len(prof1)) if list(prof3[i:i + 1 + 2 * numb_zeros_thrd_value]) == test_list3]              # search for the pattern in the profile time series
        #     t2rm3       =  np.asarray(t2rm3) + slot_strt_thrd_value
        #     for tt in t2rm3:
        #         spts2rm[tt + numb_zeros_thrd_value]  +=  (spts_track[tt + numb_zeros_thrd_value] == spt_idx)                                  # remove the spot in the isolated active frame



# class RemoveIsolatedSpots:
#     """Remove spots appearing only from an isolated frames."""
#     def __init__(self, spts_track, numb_zeros_frst_value, numb_zeros_scnd_value, numb_zeros_thrd_value, slot_strt_frst_value, slot_end_frst_value, slot_strt_scnd_value, slot_end_scnd_value, slot_strt_thrd_value, slot_end_thrd_value):
#
#         spts2rm    =  np.zeros_like(spts_track)                                                                         # initialize spots to remove matrix
#         spts_idxs  =  np.unique(spts_track[spts_track != 0])                                                            # indexes of the tracked spots
#         test_list  =  list(np.concatenate([np.zeros(numb_zeros, dtype=np.int64), np.array([1]), np.zeros(numb_zeros, dtype=np.int64)]))     # pattern of active inactive to search for in the profile
#         pbar       =  ServiceWidgets.ProgressBar(total1=spts_idxs.size)
#         pbar.show()
#
#         for cnt, spt_idx in enumerate(spts_idxs):                                                                       # for each spot
#             pbar.update_progressbar1(cnt)
#             prof  =  np.sign(np.sum(spts_track == spt_idx, axis=(1, 2)))                                                # 0/1 profile (active-inactive)
#             t2rm  =  [i for i in range(0, len(prof)) if list(prof[i:i + 1 + 2 * numb_zeros]) == test_list]              # search for the pattern in the profile time series
#             for tt in t2rm:
#                 spts2rm[tt + numb_zeros]  +=  (spts_track[tt + numb_zeros] == spt_idx)                                  # remove the spot in the isolated active frame
#         pbar.close()
#         self.spts2rm  =  spts2rm





# class OneSpotsPerNucleus:
#     """Keeps only one spot per nucleus, the biggest one."""
#     def __init__(self, spts_track):
#
#         tlen     =  spts_track.shape[0]                                                                                 # number of time steps
#         spts2rm  =  np.zeros_like(spts_track)                                                                           # initialize the matrix of the spots to remove
#         pbar     =  ServiceWidgets.ProgressBar(total1=tlen)
#         pbar.show()
#         for tt in range(tlen):
#             pbar.update_progressbar1(tt)
#             rgp       =  regionprops_table(spts_track[tt], properties=["label", "area", "area_bbox"])                   # regionprops of the tracked spots
#             sld       =  (rgp["area_bbox"] - rgp["area"]) / rgp["area"]                                                 # two spots far from each other have a bbox bigger than the sum of the bare spots
#             idxs2wrk  =  np.where(sld > 5)[0]                                                                           # when sld is higher than 1, there is something strange
#             for idx2wrk in idxs2wrk:                                                                                    # for each of the high sld spots
#                 spt      =  (spts_track[tt] == rgp["label"][idx2wrk]) * 1                                               # isolate it
#                 spt_lbl  =  label(spt, connectivity=1)                                                                  # label it
#                 if spt_lbl.max() > 1:                                                                                   # if there is more than 1 label
#                     rgp_spt   =  regionprops_table(spt_lbl, properties=["label", "area"])                               # regionprop to measure area
#                     idxs_bff  =  list(np.arange(spt_lbl.max()))                                                         # list of the indexes of the identified connected components
#                     idxs_bff.remove(np.argmax(rgp_spt["area"]))                                                         # remove from the list the index of the connectedc component with bigger surface
#                     for idx_bff in idxs_bff:
#                         spts2rm[tt]  +=  spt_lbl == rgp_spt["label"][idx_bff]                                           # add to the final matrix the spots to be removed
#         pbar.close()
#         self.spts2rm  =  spts2rm


# class OneSpotsPerNucleusOld:
#     """Keeps only one spot per nucleus, the biggest one."""
#     def __init__(self, spts_track, solidity_thr):
#
#         tlen     =  spts_track.shape[0]                                                                                 # number of time steps
#         spts2rm  =  np.zeros_like(spts_track)                                                                           # initialize the matrix of the spots to remove
#
#         pbar     =  ServiceWidgets.ProgressBar(total1=tlen)
#         pbar.show()
#
#         for tt in range(tlen):
#             pbar.update_progressbar1(tt)
#             rgp       =  regionprops_table(spts_track[tt], properties=["label", "area"])                                # regionprops of the tracked spots
#             cv_im     =  []
#             for mm in range(len(rgp["label"])):
#                 cv_im.append(np.sum(convex_hull_image((spts_track[tt] == rgp["label"][mm]).astype(np.uint8))))          # for each label calculate determine the convex hull image area
#             solidity  =  np.asarray(cv_im) / rgp["area"]                                                                # two spots far from each other have a convew hull image bigger than the sum of the bare spots
#             idxs2wrk  =  np.where(solidity > solidity_thr)[0]                                                           # for a solid spot sld = 1; when sld higher than one, the spots is split in several connected components.
#                                                                                                                         # Tuning the threshold is possible to differentiate between sister chromatids and detection errors.
#                                                                                                                         # Even for a solid single spot you can have this ration slightly higher: happens since spots are not regular
#             for idx2wrk in idxs2wrk:                                                                                    # for each of the high sld spots
#                 spt      =  (spts_track[tt] == rgp["label"][idx2wrk]) * 1                                               # isolate it
#                 spt_lbl  =  label(spt)                                                                                  # label it
#                 if spt_lbl.max() > 1:                                                                                   # if there is more than 1 label
#                     rgp_spt   =  regionprops_table(spt_lbl, properties=["label", "area"])                               # regionprop to measure area
#                     idxs_bff  =  list(np.arange(spt_lbl.max()))                                                         # list of the indexes of the identified connected components
#                     idxs_bff.remove(np.argmax(rgp_spt["area"]))                                                         # remove from the list the index of the connectedc component with bigger surface
#                     for idx_bff in idxs_bff:
#                         spts2rm[tt]  +=  spt_lbl == rgp_spt["label"][idx_bff]                                           # add to the final matrix the spots to be removed
#         pbar.close()
#         self.spts2rm  =  spts2rm

