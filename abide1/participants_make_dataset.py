"""
Make ABIDE1 participants
Read phenotypes select, save to participants
"""

import os
import pandas as pd
import numpy as np

STUDY = "abide1"
STUDY_PATH = "/neurospin/psy_sbox/{study}".format(study=STUDY)

###############################################################################
# Read Phenotypes

age_sex_dx_site = pd.read_csv(os.path.join(STUDY_PATH, 'sourcedata',
                              'Phenotypic_V1_0b.csv'), sep=',')
age_sex_dx_site = age_sex_dx_site.rename(columns={"AGE_AT_SCAN": 'age', 'SEX': 'sex', 'SITE_ID': 'site',
                                                  "SUB_ID": 'participant_id'})
age_sex_dx_site.participant_id = age_sex_dx_site.participant_id.astype(str)
age_sex_dx_site.sex = age_sex_dx_site.sex.map({1:0, 2:1}) # 1: Male, 2: Female
age_sex_dx_site['study'] = 'ABIDE1'
age_sex_dx_site['session'] = '1'
participants = age_sex_dx_site[['participant_id', 'session', 'age', 'sex', 'site', 'study']]
assert participants.participant_id.is_unique
assert participants.shape == (1112, 6)


###############################################################################
# Save participants.tsv

participants.to_csv(os.path.join(STUDY_PATH, 'participants.tsv'),
                    sep='\t', index=False)
