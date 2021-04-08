"""
Make ABIDE1 participants
Read participants, phenotypes, merge,  save to participants
"""

import os
import pandas as pd
import glob
import re
import numpy as np

STUDY = "hbn"
STUDY_PATH = "/neurospin/psy_sbox/{study}".format(study=STUDY)


###############################################################################
# Read Participants
path = "/neurospin/psy/hbn/participants.tsv"
participants = pd.read_csv(path, sep='\t')
participants.participant_id= participants.participant_id.str.replace("sub-","")


###############################################################################
# Read subjects per release in sourcedata folder
# retrieve filenames
release8_CBIC = glob.glob("/neurospin/psy/hbn/sourcedata/release8/CBIC/sub*")
release8_CUNY = glob.glob("/neurospin/psy/hbn/sourcedata/release8/CUNY/sub*")
release8_RU = glob.glob("/neurospin/psy/hbn/sourcedata/release8/RU/sub*")
release7_CBIC = glob.glob("/neurospin/psy/hbn/sourcedata/release7/CBIC/sub*")
release7_RU = glob.glob("/neurospin/psy/hbn/sourcedata/release7/RU/sub*")
release6_CBIC = glob.glob("/neurospin/psy/hbn/sourcedata/release6/CBIC/sub*")
release6_RU = glob.glob("/neurospin/psy/hbn/sourcedata/release6/RU/sub*")

# select participant_id
release6_CBIC = [[i.split("-")[1].split(".")[0], "CBIC"] for i in release6_CBIC]
release6_RU = [[i.split("-")[1].split(".")[0], "RU"] for i in release6_RU]
release7_CBIC = [[i.split("-")[1].split(".")[0], "CBIC"] for i in release7_CBIC]
release7_RU = [[i.split("-")[1].split(".")[0], "RU"] for i in release7_RU]
release8_CBIC = [[i.split("-")[1].split(".")[0], "CBIC"] for i in release8_CBIC]
release8_RU = [[i.split("-")[1].split(".")[0], "RU"] for i in release8_RU]
release8_CUNY = [[i.split("-")[1].split(".")[0], "CUNY"] for i in release8_CUNY]

# merge by release
release6_CBIC = pd.DataFrame(release6_CBIC, columns=['participant_id','site']) 
release6_RU = pd.DataFrame(release6_RU, columns=['participant_id','site'])
release6 = pd.merge(release6_CBIC, release6_RU, how='outer')
assert (release6_CBIC.shape[0]+release6_RU.shape[0]) == release6.shape[0]

release7_CBIC = pd.DataFrame(release7_CBIC, columns=['participant_id','site']) 
release7_RU = pd.DataFrame(release7_RU, columns=['participant_id','site'])
release7 = pd.merge(release7_CBIC, release7_RU, how='outer')
assert (release7_CBIC.shape[0]+release7_RU.shape[0]) == release7.shape[0]

release8_CBIC = pd.DataFrame(release8_CBIC, columns=['participant_id','site']) 
release8_RU = pd.DataFrame(release8_RU, columns=['participant_id','site'])
release8_CUNY = pd.DataFrame(release8_CUNY, columns=['participant_id','site'])
release8 = pd.merge(release8_CBIC, release8_RU, how='outer')
release8 = pd.merge(release8, release8_CUNY, how='outer')
assert (release8_CBIC.shape[0]+release8_RU.shape[0]+release8_CUNY.shape[0]) \
        == release8.shape[0]

# merge all releases
release_new = pd.merge(release6, release7, how='outer')
assert (release6.shape[0]+release7.shape[0]) == release_new.shape[0]
release_new = pd.merge(release_new, release8, how='outer')
assert (release8.shape[0]+release7.shape[0]+release6.shape[0]) \
        == release_new.shape[0]+1 
        # common subject sub-NDARYC287UFV to R7 (possible cut brain) and R8

# merge old participants with new
new_participants = pd.merge(participants, release_new,
                            on=['participant_id', 'site'],
                            how='outer', sort=False, validate='1:1')


###############################################################################
# Read Phenotypes
# to add age and sex
root = "/neurospin/psy_sbox/hbn/sourcedata"
sourcedata = ["HBN_R4_Pheno.csv", "HBN_R7_Pheno.csv",
              "HBN_R2_1_Pheno.csv", "HBN_R5_Pheno.csv", "HBN_R8_Pheno.csv",
              "HBN_R3_Pheno.csv", "HBN_R6_Pheno.csv", "HBN_R9_Pheno.csv"]
#init    
release_all = pd.read_csv(os.path.join(root,"HBN_R1_1_Pheno.csv"))
release_all = release_all.rename(columns={'Age': 'age', 'Sex': 'sex',
                                          'EID': 'participant_id'})
release_all = release_all [['participant_id', 'age', 'sex']]
s = release_all.shape[0]

# merge phenotypes
for i in sourcedata:
    release_x = pd.read_csv(os.path.join(root,i))
    release_x = release_x.rename(columns={'Age': 'age', 'Sex': 'sex',
                                          'EID': 'participant_id'})
    release_x = release_x [['participant_id', 'age', 'sex']]
    release_x = release_x.drop_duplicates(subset=['participant_id'])
    s += release_x.shape[0]
    assert release_x.participant_id.is_unique
    release_all = pd.merge(release_all, release_x, 
                           on=['participant_id', 'age', 'sex'],
                           how='outer', sort=False, validate='1:1')
    #print(release_all.shape[0], s) # common subjects un pheno csv
assert release_all.participant_id.is_unique
release_all.age = release_all.age.astype('str')
release_all.sex = release_all.sex.astype('str')
release_all.sex = release_all.sex.str.replace(".0","")

# merge participants and phenotypes
new_participants = pd.merge(new_participants, release_all,
                            on='participant_id', how='outer')
new_participants = new_participants[['participant_id', 'age', 'sex', 'site']]
assert new_participants.participant_id.is_unique

# not all subjects have an MRI, we remove subject with EEG only
# to do so : unselect Nan site
new_participants = new_participants.loc[new_participants['site'] \
                                        == new_participants['site']]
new_participants['study']="HBN"
assert new_participants.shape[0]== 2500


###############################################################################
# Save participants.tsv

new_participants.to_csv(os.path.join(STUDY_PATH, 'participants.tsv'),
                    sep='\t', index=False)


