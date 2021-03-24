"""
Make LOCALIZER cat12vbm dataset
Read participants, merge with TIV and save to phenotype
"""

import sys
sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
from utils import cat12_nii2npy
import os
import pandas as pd

STUDY = "localizer"
STUDY_PATH = "/neurospin/psy_sbox/{study}".format(study=STUDY)
OUTPUT_DIR = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"
N_SUBJECTS = 88

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
assert tiv.shape[0] == N_SUBJECTS


###############################################################################
## Merge with TIV and ROIs

age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')
phenotype_pd = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert tiv.shape[0] == N_SUBJECTS
assert len(phenotype_pd) == len(tiv) - 4

###############################################################################
# Input nii files

nii_regex_path = '/neurospin/psy/{study}/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii'.format(study=STUDY)

## Add the QC file if available
qc_path = '/neurospin/psy_sbox/{study}/derivatives/cat12-12.6_vbm_qc/qc.tsv'.format(study=STUDY)

participants_filename, rois_filename, vbm_filename =\
    cat12_nii2npy(nii_path=nii_regex_path, phenotype=phenotype_pd, dataset=STUDY,
                  output_path=OUTPUT_DIR, qc=qc_path, sep='\t', id_type=str,
                check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))


###############################################################################
# QC relaod npy, ROIs, check n_participants

participants = pd.read_csv(participants_filename)
rois = pd.read_csv(rois_filename)
imgs_arr = np.load(vbm_filename)
mask_img = nibabel.load(os.path.join(output, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))

assert participants.shape[0] == N_SUBJECTS
assert rois.shape[0] == N_SUBJECTS
assert imgs_arr.shape == (N_SUBJECTS, 1, 121, 145, 121)

