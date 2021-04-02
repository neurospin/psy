"""
Make ABIDE1 phenotype
Read participants, phenotypes merge with TIV and save to phenotype
"""

import os
import sys
import pandas as pd
import numpy as np
sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
from utils import df_column_switch

STUDY = "abide1"
STUDY_PATH = "/neurospin/psy_sbox/{study}".format(study=STUDY)

###############################################################################
# Read Phenotypes

age_sex_dx_site = pd.read_csv(os.path.join(STUDY_PATH, 'sourcedata',
                              'Phenotypic_V1_0b.csv'), sep=',')
age_sex_dx_site = age_sex_dx_site.rename(columns={"AGE_AT_SCAN": 'age', 'SEX': 'sex', 'SITE_ID': 'site',
                                                  "DX_GROUP": "diagnosis", "SUB_ID": 'participant_id'})
age_sex_dx_site.diagnosis = age_sex_dx_site.diagnosis.map({1: 'autism', 2:'control'})
age_sex_dx_site.participant_id = age_sex_dx_site.participant_id.astype(str)
age_sex_dx_site.sex = age_sex_dx_site.sex.map({1:0, 2:1}) # 1: Male, 2: Female
age_sex_dx_site['study'] = 'ABIDE1'
age_sex_dx_site = df_column_switch(age_sex_dx_site, "site", "age")
age_sex_dx_site = df_column_switch(age_sex_dx_site, "diagnosis", "sex")
age_sex_dx_site = df_column_switch(age_sex_dx_site, "DSM_IV_TR", "diagnosis")
age_sex_dx_site = df_column_switch(age_sex_dx_site, "site", "diagnosis")
assert age_sex_dx_site.participant_id.is_unique
assert age_sex_dx_site.shape == (1112, 75)

###############################################################################
# Read TIV

tiv = pd.read_csv(os.path.join(STUDY_PATH, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv = tiv.rename(columns={"TIV": 'tiv'})
tiv.participant_id = tiv.participant_id.astype(str)
assert tiv.shape[0] == 1098

###############################################################################
## Merge with TIV and ROIs

age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='m:1')

# split in  2 files : rois and participants info
tiv_columns = list(tiv.columns)[2:]

age_sex_dx_site_study = age_sex_dx_site_study_tiv.drop(tiv_columns, axis=1)
rois = age_sex_dx_site_study_tiv[tiv.columns]

assert rois.shape == (1098, 290)
assert age_sex_dx_site_study.shape == (1098, 76)

###############################################################################
# Save

age_sex_dx_site_study.to_csv(os.path.join(STUDY_PATH,"phenotype",
                                          'participants_diagnosis.tsv'),
                             sep='\t', index=False)

rois.to_csv(os.path.join(STUDY_PATH,"phenotype",
                                          'participants_ROIS.tsv'),
                             sep='\t', index=False)
