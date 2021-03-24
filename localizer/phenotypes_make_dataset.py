"""
Make LOCALIZER phenotype
Read participants, merge with TIV and save to phenotype
"""

import os
import pandas as pd

LOCALIZER_PATH = "/neurospin/psy_sbox/localizer"

###############################################################################
# Read Participants

age_sex_dx_site = pd.read_csv(os.path.join(LOCALIZER_PATH, 'participants.tsv'), sep='\t')
age_sex_dx_site.sex = age_sex_dx_site.sex.map({'M':0, 'F':1})
age_sex_dx_site['study'] = 'LOCALIZER'
age_sex_dx_site.age = age_sex_dx_site.age.astype(float, errors='ignore')
age_sex_dx_site['diagnosis'] = 'control'
assert age_sex_dx_site.participant_id.is_unique
assert age_sex_dx_site.shape == (94, 6)

###############################################################################
# Read TIV

tiv = pd.read_csv(os.path.join(LOCALIZER_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
assert tiv.shape[0] == 88

###############################################################################
## Merge with TIV and ROIs

age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')
participants_columns_list = list(age_sex_dx_site.columns)
participants_columns_list = [participants_columns_list[0]] + ["session", "run"] + \
    participants_columns_list[1:]
age_sex_dx_site_study = age_sex_dx_site_study_tiv[participants_columns_list]

rois = age_sex_dx_site_study_tiv[tiv.columns]

assert rois.shape == (88, 291)
assert age_sex_dx_site_study.shape == (88, 8)

###############################################################################
# Save

age_sex_dx_site_study.to_csv(os.path.join(LOCALIZER_PATH,"phenotype",
                                          'participants_diagnosis.tsv'),
                             sep='\t', index=False)

rois.to_csv(os.path.join(LOCALIZER_PATH,"phenotype",
                                          'participants_ROIS.tsv'),
                             sep='\t', index=False)
