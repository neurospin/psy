"""
Make LOCALIZER cat12vbm dataset
Read participants, merge with TIV and save to phenotype
"""

import sys
import os
import numpy as np
import pandas as pd
import nibabel

sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
from utils import cat12_nii2npy, diff_sets, ml_regression

STUDY = "localizer"
STUDY_PATH = "/neurospin/psy_sbox/{study}".format(study=STUDY)
OUTPUT_DIR = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"
N_SUBJECTS = 81

###############################################################################
# Read Participants

age_sex_dx_site = pd.read_csv(os.path.join(STUDY_PATH, 'participants.tsv'), sep='\t')
age_sex_dx_site.sex = age_sex_dx_site.sex.map({'M':0, 'F':1})
age_sex_dx_site['study'] = STUDY
age_sex_dx_site = age_sex_dx_site[~age_sex_dx_site.age.isna() & ~age_sex_dx_site.age.eq('None')] # 4 participants have 'None' age
age_sex_dx_site.age = age_sex_dx_site.age.astype(float)
age_sex_dx_site['diagnosis'] = 'control'
assert age_sex_dx_site.participant_id.is_unique
assert age_sex_dx_site.shape == (90, 6)


###############################################################################
# Read TIV and ROIs

tiv = pd.read_csv(os.path.join(STUDY_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
if 'TIV' in tiv:
    tiv.rename(columns={'TIV': 'tiv'}, inplace=True)

assert tiv.shape[0] == 88


###############################################################################
## Merge with TIV and ROIs

age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')
phenotype = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                      ~age_sex_dx_site_study_tiv.tiv.isna() &
                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(phenotype) == len(tiv) - 4
assert phenotype.shape[0] == 84

###############################################################################
# Input nii files

nii_path = '/neurospin/psy/{study}/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii'.format(study=STUDY)

## Add the QC file if available
qc_path = '/neurospin/psy_sbox/{study}/derivatives/cat12-12.6_vbm_qc/qc.tsv'.format(study=STUDY)
qc = pd.read_csv(qc_path, sep='\t') if qc_path is not None else None

participants_filename, rois_filename, vbm_filename =\
    cat12_nii2npy(nii_path=nii_path, phenotype=phenotype, dataset=STUDY,
                  output_path=OUTPUT_DIR, qc=qc, sep='\t', id_type=str,
                  check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))


###############################################################################
# QC reload npy, ROIs, check n_participants

participants = pd.read_csv(participants_filename, sep='\t')
rois = pd.read_csv(rois_filename, sep='\t')
imgs_arr = np.load(vbm_filename)

assert participants.shape[0] == N_SUBJECTS
assert rois.shape[0] == N_SUBJECTS
assert imgs_arr.shape == (N_SUBJECTS, 1, 121, 145, 121)

###############################################################################
# Differences with Benoit

benoit_participants_filename = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/localizer_t1mri_mwp1_participants.csv'

if os.path.exists(benoit_participants_filename):
    participants_benoit = pd.read_csv(benoit_participants_filename)
    diff, in_benoit, in_new = diff_sets(participants_benoit.participant_id, participants.participant_id)
    assert {'S74'} == diff == in_benoit
    # Lost one subject 'S74' because of QC


###############################################################################
# Basic QC predict age

mask_img = nibabel.load(os.path.join(OUTPUT_DIR, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
mask_arr = mask_img.get_fdata() != 0
data = dict(rois=rois.loc[:, 'l3thVen_GM_Vol':].values,
            vbm=imgs_arr.squeeze()[:, mask_arr])
y = participants['age'].values

res_age = ml_regression(data, y)

# Excpected results
expected = pd.DataFrame(
    [["rois",  0.198864,  4.106239,  5.808097],
     ["vbm",   0.118965,  4.401617,  6.222352]],
    columns = ["data",        "r2",       "mae",      "rmse"]
)

from pandas._testing import assert_frame_equal
assert_frame_equal(res_age, expected)
