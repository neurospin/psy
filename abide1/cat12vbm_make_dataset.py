"""
Make ABIDE1 cat12vbm dataset
Read phenotypes, merge with TIV and save to arrays
"""

import sys
import os
import numpy as np
import pandas as pd
import nibabel
sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
from utils import cat12_nii2npy, diff_sets, ml_regression, ml_correlation_plot

STUDY = "abide1"
STUDY_PATH = "/neurospin/psy_sbox/{study}".format(study=STUDY)
OUTPUT_DIR = "/neurospin/psy_sbox/all_studies/derivatives/arrays"
N_SUBJECTS = 1082

###############################################################################
# Read Phenotypes

age_sex_dx_site = pd.read_csv(os.path.join(STUDY_PATH, 'phenotype',
                              'participants_diagnosis.tsv'), sep='\t')
age_sex_dx_site.participant_id = age_sex_dx_site.participant_id.astype(str)
age_sex_dx_site.session = age_sex_dx_site.session.astype(str)
assert age_sex_dx_site.participant_id.is_unique
assert age_sex_dx_site.shape == (1098, 76)

###############################################################################
# Read TIV and ROIs

tiv = pd.read_csv(os.path.join(STUDY_PATH, 'phenotype',
                              'participants_ROIS.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
tiv.session = tiv.session.astype(str)
assert tiv.shape[0] == 1098

###############################################################################
# Keep ROIs and phenotypes columns

tiv_columns = list(tiv.columns)[2:]
participants_columns = list(age_sex_dx_site.columns)[2:]

###############################################################################
## Merge with TIV and ROIs

age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on=['participant_id', 'session'], how='left', sort=False, validate='m:1')
phenotype = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.tiv.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna() &
                                                      ~age_sex_dx_site_study_tiv.site.isna()
                                                      ]
assert len(phenotype) == len(tiv) - 1 # No TIV available for participant 50818
assert phenotype.shape[0] == 1097

###############################################################################
# Input nii files

nii_path = '/neurospin/psy_sbox/{study}/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii'.format(study=STUDY)

## Add the QC file if available
qc_path = '/neurospin/psy_sbox/{study}/derivatives/cat12-12.6_vbm_qc/qc.tsv'.format(study=STUDY)
qc = pd.read_csv(qc_path, sep='\t') if qc_path is not None else None

participants_filename, rois_filename, vbm_filename =\
    cat12_nii2npy(nii_path=nii_path, phenotype=phenotype, dataset=STUDY,
                  output_path=OUTPUT_DIR, qc=qc, sep='\t', id_type=str,
                  check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)),
                  tiv_columns=tiv_columns, participants_columns=participants_columns)


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

benoit_participants_filename = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/abide1_t1mri_mwp1_participants.csv'

if os.path.exists(benoit_participants_filename):
    participants_benoit = pd.read_csv(benoit_participants_filename)
    diff, in_benoit, in_new = diff_sets(participants_benoit.participant_id, participants.participant_id)
    assert {50274, 50278, 50312, 50313, 50345, 50317, 50322, 50136, 51160, 51581} == diff
    assert {50317} == in_benoit
    assert {50274, 50278, 50312, 50313, 50345, 50322, 51160, 50136, 51581} == in_new
    # Lost one subject '50317' because of QC.
    # Add the other because of modified QC rule (keep all 1)


###############################################################################
# Basic QC predict age

mask_img = nibabel.load(os.path.join(OUTPUT_DIR, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
mask_arr = mask_img.get_fdata() != 0
data = dict(rois=rois.loc[:, 'l3thVen_GM_Vol':].values,
            vbm=imgs_arr.squeeze()[:, mask_arr])
y = participants['age'].values

res_age = ml_regression(data, y)


# Expected results
expected = pd.DataFrame(
    [["rois",  0.246942,  3.367512,  6.541419],
     ["vbm",   0.771252,  2.644638,  3.822328]
],
    columns = ["data",        "r2",       "mae",      "rmse"]
)

from pandas.testing import assert_frame_equal
assert_frame_equal(res_age, expected)

# Correlation Plot
ml_correlation_plot(data, y, OUTPUT_DIR, STUDY)