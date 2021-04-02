"""
Make ABIDE1 cat12vbm QC
"""
import sys
sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
import pandas as pd
import os
import argparse
import nibabel
from cat12vbm_qc_utils import parse_xml_files_scoresQC
from cat12vbm_qc_utils import plot_pca, compute_mean_correlation,\
                                pdf_plottings, pdf_cat, mwp1toreport,\
                                concat_tsv, pdf_cat2
from cat12vbm_qc_utils import img_to_array
from cat12vbm_qc_utils import compute_brain_mask


def launch_cat12_qc(img_filenames, mask_filenames, root_cat12vbm, inputscores):
    # retrieve qcscores
    root_qc = root_cat12vbm+"_qc"
    if (not os.path.exists(root_qc)):
        root_psy_sbox = root_cat12vbm.replace("/psy/", "/psy_sbox/")
        root_qc = root_psy_sbox+"_qc"
    output_scores = os.path.join(root_qc, "scoresQC.tsv")
    parse_xml_files_scoresQC(inputscores, output_scores)

    # correlation
    imgs_arr, df, ref_img = img_to_array(img_filenames)

    if mask_filenames is None:
        mask_img = compute_brain_mask(imgs_arr, ref_img)
        mask_arr = mask_img.get_fdata() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) == 1:
        mask_img = nibabel.load(mask_filenames[0])
        mask_arr = mask_img.get_fdata() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) > 1:
        assert len(mask_filenames) == len(imgs_arr), "The list of .nii masks must have the same length as the " \
                                                     "list of .nii input files"
        mask_glob = [nibabel.load(mask_filename).get_fdata()>0 for mask_filename in mask_filenames]
        imgs_arr = imgs_arr.squeeze()[mask_glob]

    # PCA
    plot_pca(imgs_arr, df)
    # MEAN CORR
    mean_corr = compute_mean_correlation(imgs_arr, df)
    path_corr = os.path.join(root_qc, "mean_cor.tsv")
    mean_corr.to_csv(path_corr, index=False, sep='\t')
    # MEAN CORR AND QC SCORES
    qc_table = concat_tsv(path_corr, output_scores)
    path_qc = os.path.join(root_qc, "qc.tsv")
    qc_table.to_csv(path_qc, index=False, sep='\t')
    # nii brain images pdf ordored by mean correlation
    mean_corr = mean_corr.values
    niipdf = os.path.join(root_qc, 'nii_plottings.pdf')
    nii_filenames_sorted = [df[df['participant_id'].eq(id)].path.values[0] for (id, _, _, _) in mean_corr]
    pdf_plottings(nii_filenames_sorted, mean_corr, niipdf, limit=None)
    # cat12vbm reports pdf ordored by mean correlation
    reportpdf = os.path.join(root_qc, 'cat12_reports.pdf')
    nii_filenames_pdf = mwp1toreport(nii_filenames_sorted, root_cat12vbm)
    pdf_cat2(nii_filenames_pdf, reportpdf, 300)

    return 0

def apply_qc_limit_criteria(study_path, root_cat12vbm):
    root_qc = root_cat12vbm+"_qc"
    if not os.path.exists(root_qc):
        root_psy_sbox = root_cat12vbm.replace("/psy/", "/psy_sbox/")
        root_qc = root_psy_sbox+"_qc"
    participants = pd.read_csv(os.path.join(study_path, "Phenotypic_V1_0b.csv"), sep=',')
    participants = participants.rename(columns={"AGE_AT_SCAN": 'age', 'SEX': 'sex', 'SITE_ID': 'site',
                                                  "DX_GROUP": "diagnosis", "SUB_ID": 'participant_id'})
    participants.participant_id = participants.participant_id.astype(str)

    qc = pd.read_csv(os.path.join(root_qc, 'qc.tsv'), sep= "\t")
    qc.participant_id = qc.participant_id.astype(str)
    qc.corr_mean[ qc.corr_mean.abs() > 1] = qc.corr_mean.median()
    qc["qc"] = 1
    qc.loc[(qc.NCR > 4.5) | (qc.IQR > 4.5), "qc"] = 0

    csv_filename = os.path.join(root_qc, 'qc.tsv')
    print("Save to %s" % csv_filename)
    print("Perform manual look at the data, manually discard (set qc=0) participants in %s" % csv_filename)
    qc.to_csv(csv_filename, sep='\t', index=False)

    # participant not processed with cat12vbm
    nocat12_filename = os.path.join(root_qc, 'noprocessed_participants.tsv')
    nocat12 = participants.participant_id[~participants.participant_id.isin(qc.participant_id)]
    nocat12 = pd.DataFrame(nocat12, columns=['participant_id', 'err'])

    for index, row in nocat12.iterrows():
        sub = row['participant_id']
        err = "sub-{0}/ses-1/anat/err".format(sub)
        path_err = os.path.join(root_cat12vbm, err)
        if os.path.exists(path_err):
            err_name = os.listdir(path_err)[0]
            err_name = err_name.split(".")[-1]
            row["err"] = err_name
        else:
            print("this subject is not in cat12vbm : {0}".format(sub))

    nocat12.to_csv(nocat12_filename, sep='\t', index=False, header=['participant_id', 'err'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='A list of .nii files', required=True, nargs='+', type=str)
    parser.add_argument('--mask', help='A list of .nii masks or a single .nii mask', default=None, nargs='+', type=str)
    parser.add_argument('--input_qcscores', help='A list of .xml files', required=True, nargs='+', type=str)
    parser.add_argument('--root_cat12vbm', help='root to cat12vbm', required=True, nargs=1, type=str)
    parser.add_argument('--output_nii_pdf', help='The output path to the .pdf file', nargs=1, default='nii_plottings.pdf', type=str)
    parser.add_argument('--output_pdf', help='The output path to the .pdf file', nargs=1, default='cat12vbm_reports.pdf', type=str)
    parser.add_argument('--limit', help='The max number of slice to plot', default=None, type=int)

    options = parser.parse_args()
    # inputs
    img_filenames = options.input
    mask_filenames = options.mask
    input_qcscores = options.input_qcscores
    # paths
    root_cat12vbm = options.root_cat12vbm[0]
    study_path = root_cat12vbm.split(os.sep)[0:-2]
    study_path = os.sep.join(study_path)

    # launch_cat12_qc(img_filenames, mask_filenames, root_cat12vbm, input_qcscores)
    apply_qc_limit_criteria(study_path, root_cat12vbm)

    # COMMAND Terminal
    # python3 /neurospin/psy_sbox/git/ns-datasets/abide1/cat12vbm_qc.py --input /neurospin/psy_sbox/abide1/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1sub[-_0-9a-zA-Z]*_T1w.nii --input_qcscores /neurospin/psy_sbox/abide1/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/report/cat_sub-*_T1w.xml --root_cat12vbm /neurospin/psy_sbox/abide1/derivatives/cat12-12.6_vbm

if __name__ == "__main__":
    main()
