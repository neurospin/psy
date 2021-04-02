"""
pynet data preprocessing : n processes script
========================
Script to launch a multi-processing pre-processing pipeline
Based on cat12vbm masks

Pipeline details:
    skullstripped with cat12vbm masks
    reorient to standard
    scale : 1 iso
    Correction of biasfield with ants
    Linear registration to MNI template
    Noises around the brain corrections (caused by linear regitration)


"""

import os
import sys

import re
import subprocess
import nibabel
import argparse
import numpy as np
import itertools
import glob

from pynet.preprocessing import reorient2std
from pynet.preprocessing import scale
from pynet.preprocessing import register
from pynet.preprocessing import EraseNoise
from pynet.preprocessing import biasfield
from pynet.preprocessing import brainmask
from pynet.preprocessing import Processor
from nitk.bids.bids_utils import get_keys

from threading import Thread, RLock

verrou = RLock()


class Preprocessing_dl(Thread):

    """ Thread in charge of the preprocessing of a set of images """

    def __init__(self, list_images, list_already_done, dest_path, maskscat12):
        Thread.__init__(self)
        self.list_images = list_images
        self.list_masks = maskscat12
        self.list_already_done = list_already_done
        self.dest_path = dest_path

    def run(self):
        """ Code to execute during thread execution """
        for c, (file, root) in enumerate(self.list_images):
            # execute pipeline
            print("\nthe file processed is : ", file)
            path_image = os.path.join(root, file)
            image = nibabel.load(path_image)
            mask = nibabel.load(self.list_masks[c])
            target = nibabel.load('/i2bm/local/fsl/data/standard'
                                  '/MNI152_T1_1mm_brain.nii.gz')
            pipeline = Processor()
            pipeline.register(brainmask, mask=mask,
                              check_pkg_version=False,
                              apply_to="image")
            pipeline.register(reorient2std, check_pkg_version=False,
                              apply_to="image")
            pipeline.register(scale, scale=1, check_pkg_version=False,
                              apply_to="image")
            pipeline.register(biasfield, check_pkg_version=False,
                              apply_to="image")
            pipeline.register(register, target=target,
                              check_pkg_version=False,
                              apply_to="image")
            pipeline.register(EraseNoise, check_pkg_version=False,
                              apply_to="image")

            normalized = pipeline(image)

            # write the results in bids format
            path_tmp = path_image.split(os.sep)
            filename = path_tmp[-1]
            newfilename = filename.split("_")
            if re.search("ses-", filename):
                newfilename.insert(2, "preproc-linear")
            else:
                raise ValueError("session key is needed")
            newfilename = "_".join(newfilename)
            ses = path_tmp[-3]
            sub = path_tmp[-4]
            sub_dest = os.path.join(self.dest_path, sub)
            ses_dest = os.path.join(sub_dest, ses)
            anatdest = os.path.join(ses_dest, "anat")
            #create filetree
            subprocess.check_call(['mkdir', '-p', anatdest])
            if re.search(".gz", file):
                end_path = "/{0}/{1}/anat/{2}"\
                           .format(sub, ses, newfilename)
            else:
                end_path = "/{0}/{1}/anat/{2}.gz"\
                           .format(sub, ses, newfilename)
            dest_file = self.dest_path+end_path
            # save results
            nibabel.save(normalized, dest_file)
            with verrou:
                # write already preprocessed images
                already_done = os.path.join(self.dest_path, "already_done.txt")
                with open(already_done, "a") as file1:
                    ligne = file+"\n"
                    file1.write(ligne)


def read_alreadydone(dest_path):
    """Read already preprocessed images file.

    Parameters
    ----------
    dest_path: string
        path to the output.

    Returns
    -------
    list_already_done: list
        list of the already preprocessed image.
    """
    list_already_done = []
    already_done = os.path.join(dest_path, "already_done.txt")
    if os.path.exists(already_done):
        with open(already_done, "r") as file1:
            for line in file1.readlines():
                list_already_done.append(line[0:-1])
    else:
        file_object = open(already_done, "x")
        file_object.close()
    return list_already_done


def divise_namelist(list1, number_process):
    """Divides images to preprocessed into n batches.

    Parameters
    ----------
    path_rawdata: string
        path to the rawdata folder.
    number_process: int
        number of processes to launch
    list_already_done: list
        list of the already preprocessed images.

    Returns
    -------
    out: list of list
        list of batches to launch.
    """
    avg = len(list1)/float(number_process)
    out = []
    last = 0.0
    while last < len(list1):
        out.append(list1[int(last):int(last+avg)])
        last += avg
    return out

def masklistordered(path_rawdata, root_cat12, list_already_done):
    """Create an ordered list of cat12vbm p0 mask.

    Parameters
    ----------
    path_rawdata: string
        path to the rawdata folder.
    root_cat12: string
        path to cat12vbm folder
    list_already_done: list
        list of the already preprocessed images.

    Returns
    -------
    out: list of list
        list of batches to launch.
    """
    # check if there is average images in derivatives
    list_avr = glob.glob("{0}/sub-*/ses-*/anat/*average*T1w*.nii".format(root_cat12))
    # check if session folder exists
    if not glob.glob("{0}/sub-*/ses-*".format(root_cat12)):
        raise ValueError("session folder is needed")
    # check if anat folder exists
    if not glob.glob("{0}/sub-*/ses-*/anat".format(root_cat12)):
        raise ValueError("anat folder is needed")
    liste_mask = []
    list1 = []
    liste_index = []
    liste_done = []
    for root, dirs, files in os.walk(path_rawdata):
        for file in files:
            if re.search("T1w[_]*[a-zA-Z0-9]*.nii", file)\
               and file not in list_already_done\
               and not re.search("err", root):
                file_keys = get_keys(file)
                runid = "run-{0}".format(file_keys['run'])
                if list_avr:
                    # averages are in derivatives and not in rawdata
                    runavg = file.replace(runid, "run-average")
                    # add only one time the run-average to the list
                    if runavg not in list_already_done and runavg not in liste_done:
                        liste_done.append(runavg)
                        list1.append([file, root])
                else:
                    list1.append([file, root])
    for c1, (file, root) in enumerate(list1):
        file_keys = get_keys(file)
        participant_id = file_keys['participant_id']
        session = file_keys['session']
        run = file_keys['run']
        # mask root
        root_mask = os.path.join(root_cat12, "sub-{0}/ses-{1}/anat/mri".format(participant_id, session))
        # mask name
        if re.search("nii.gz", file):
            mask_filename = "p0{0}".format(file)[0:-3]
        else:
            mask_filename = "p0{0}".format(file)
        if re.search("run", file):
            runid = "run-{0}".format(run)
            if list_avr:
                # to do only if avg
                mask_filename_run_average = mask_filename.replace(runid, "run-average")
            else:
                mask_filename_run_average = "None"
        # mask path
        mask_path = os.path.join(root_mask, mask_filename)
        # select element of the list with cat12vbm mask
        if list_avr:
            maskaverpath = os.path.join(root_mask, mask_filename_run_average)
            if (os.path.isfile(mask_path)) and (os.path.isfile(maskaverpath)) and (mask_path != maskaverpath):
                liste_mask.append(mask_path)
            elif (os.path.isfile(mask_path)) and (not os.path.isfile(maskaverpath)):
                liste_mask.append(mask_path)
            elif (not os.path.isfile(mask_path)) and (os.path.isfile(maskaverpath)):
                liste_mask.append(maskaverpath)
                list1[c1][0] = list1[c1][0].replace(runid, "run-average")
                root_avr = root_mask.split("/")[:-1]
                list1[c1][1] = "/".join(root_avr)
            else:
                liste_index.append(c1)
        else:
            if (os.path.isfile(mask_path)):
                liste_mask.append(mask_path)
            else:
                liste_index.append(c1)

    # delete element of the list whithout cat12vbm mask
    list1 = np.delete(list1, liste_index, 0).tolist()
    new_list = []
    index = []
    # remove duplicates
    for c2, elem in enumerate(list1):
        if elem not in new_list:
            new_list.append(elem)
        else:
            index.append(c2)
    liste_mask = np.delete(liste_mask, index, 0).tolist()

    return liste_mask, new_list



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--rawdata', help='path to rawdata', nargs='+', required=True, type=str)
    parser.add_argument('--maskcat12', help='path to mask', nargs='+', required=True, type=str)
    parser.add_argument('-o', '--output', help='path to quasi-raw output', nargs='+', required=True, type=str)
    parser.add_argument('-j', help='number of threads', nargs='+', required=True, type=int)
    options = parser.parse_args()

    if options.rawdata is None:
        parser.print_help()
        raise SystemExit("Error: Rawdata is missing.")

    if options.maskcat12 is None:
        parser.print_help()
        raise SystemExit("Error: Rawdata is missing.")

    if options.output is None:
        parser.print_help()
        raise SystemExit("Error: Output is missing.")

    if options.j is None:
        parser.print_help()
        raise SystemExit("Error: Number of threads is missing.")


    # initialization
    path_rawdata = options.rawdata[0]
    root_cat12 = options.maskcat12[0]
    dest_path = options.output[0]
    number_process = options.j[0]

    list_already_done = read_alreadydone(dest_path)

    ## launch n processes

    # create rawdata list and masks list from rawdata list
    maskscat12, biglist = masklistordered(path_rawdata, root_cat12, list_already_done)

    # split rawdata liste
    biglist = divise_namelist(biglist, number_process)
    # split masks list
    maskscat12 = divise_namelist(maskscat12, number_process)

    # check if mask list correspond to the rawdata list
    if len(maskscat12) != len(biglist):
        raise ValueError("number of masks is not the same than nii files")


    # check if there are any unprocessed images left
    if len(biglist) > 0:
        # check if each rawdata image has a corresponding mask
        if any([i for i in maskscat12]):
            # threads name list
            processes_list = []
            processes_obj_list = []
            for i in range(number_process):
                processes_list.append("thread_"+str(i))
            # print(processes_list)
            # threads creation
            for c, j in enumerate(processes_list):
                if len(biglist[c]) > 0:
                    j = Preprocessing_dl(biglist[c], list_already_done, dest_path, maskscat12[c])
                    j.name = processes_list[c]
                    processes_obj_list.append(j)
                else:
                    print("empty thread")
            # threads launch
            for j in processes_obj_list:
                j.start()
            # waiting for threads to finish
            for j in processes_obj_list:
                j.join()
        else:
            print("no cat12vbm mask found")
    else:
        print("no more T1w to processed")

if __name__ == "__main__":
    main()
