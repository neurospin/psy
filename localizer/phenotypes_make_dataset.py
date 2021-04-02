"""
Make LOCALIZER phenotype
Read participants, merge with TIV and save to phenotype
"""

import os
import pandas as pd

STUDY = "localizer"
STUDY_PATH = "/neurospin/psy_sbox/{study}".format(study=STUDY)

###############################################################################
# Read Participants

age_sex_dx_site = pd.read_csv(os.path.join(STUDY_PATH, 'participants.tsv'), sep='\t')
age_sex_dx_site.sex = age_sex_dx_site.sex.map({'M':0, 'F':1})
age_sex_dx_site['study'] = 'LOCALIZER'
age_sex_dx_site.age = age_sex_dx_site.age.astype(float, errors='ignore')
age_sex_dx_site['diagnosis'] = 'control'
assert age_sex_dx_site.participant_id.is_unique
assert age_sex_dx_site.shape == (94, 6)

###############################################################################
# Read TIV

tiv = pd.read_csv(os.path.join(STUDY_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
assert tiv.shape[0] == 88

###############################################################################
## Merge with TIV and ROIs

age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')

# split in  2 files : rois and participants info
tiv_columns = list(tiv.columns)[3:]

age_sex_dx_site_study = age_sex_dx_site_study_tiv.drop(tiv_columns)
rois = age_sex_dx_site_study_tiv[tiv.columns]

assert rois.shape == (88, 291)
assert age_sex_dx_site_study.shape == (88, 8)

###############################################################################
# Save

age_sex_dx_site_study.to_csv(os.path.join(STUDY_PATH,"phenotype",
                                          'participants_diagnosis.tsv'),
                             sep='\t', index=False)

rois.to_csv(os.path.join(STUDY_PATH,"phenotype",
                                          'participants_ROIS.tsv'),
                             sep='\t', index=False)
