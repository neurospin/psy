[CAT12](http://www.neuro.uni-jena.de/cat12-html/)

# Quality Control (QC)

Outputs: `cat12-<version>_vbm_qc/`

- `qc.tsv`: contains `participant_id`, `corr_mean` with others participants.
   The file is sorted by increasing `corr_mean`, providing outliers first.
   `NCR`, `ICR`, `IQR` QC metrics see below.
  `qc` in [0, 1]

- `pca.pdf`: PCA on GM mapt to identify outliers

- `nii_plottings.pdf`: GM map sorted by increasing `corr_mean`, providing outliers first.


[CAT report and quality control (in development)](http://www.neuro.uni-jena.de/cat12-html/cat_methods_QA.html)

- NCR (Noise Contrast Ratio): The NCR measures the local standard deviation in the optimized WM segment and is scaled by the minimum tissue contrast.

- ICR (Inhomogeneity Contrast Ratio): The ICR is measured as the global standard deviation within the optimized WM segment and is scaled by the minimum tissue contrast.

- IQR (image quality rating) The resulting ratings were combined as weighted average image quality rating IQR (Dahnke et al. 2016).

The obtained quality ratings range from 0.5 (100 rating points (rps)) to 10.5 (0 rps) with values around 1 and 2 describing (very) good image quality (grad A and B) and values around 5 (grad E) and higher (grad F, less than 50 rps) indicating problematic images

Default strategy:

1. Discard (`qc` column of `qc.tsv`) participants with <IQR or NCR> see larger than 4.5
2. Inspect image (`nii_plottings.pdf`) by increasing `corr_mean` and manually discard participants.

Catreport content :
http://www.neuro.uni-jena.de/cat12-html/cat_methods_catreporthelp.html
http://www.neuro.uni-jena.de/cat12-html/cat_methods_errormanagement.html


[In Finkelmeyer et al. 2018](https://www.sciencedirect.com/science/article/pii/S221315821730236X):
_The toolbox further provided ratings of image data quality, which were used to identify problems with individual images. These assess basic image properties, noise and geometric distortions (e.g. due to motion) and combine them into a weighted image quality rating (IQR)._

