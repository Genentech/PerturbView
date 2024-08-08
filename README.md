# PerturbView Tissue Analysis

Code for running base calling for PerturbView in tissue, and aligning tissue data to Xenium Spatial Transcriptomics.

## Installation
Tested on python 3.10.13.  Later versions of python may not work

install the requirements in a virtualenv or conda-env with:
```
pip install -r requirements.txt
```

The spotiflow `general` model must also be copied to `./models/general/`

## Usage
The pipeline is currently run through a snakemake script.  Please edit the `snakefile` to point to relevant paths for your experiment:

`sub_dir` = this is the name of your experiment

`sample_dir` = location of your ISS imaging.  We store this in `{sub_dir}/ISS`

`pheno_dir` = location of your phenotyping imaging (antibodies, stains, or Xenium DAPI image).  We store this in `{sub_dir}/pheno`

For calling gRNAs or other barcodes via ISS ensure that a CSV is filed set to the `grnas` variable.  The file should at minimum contain a column called either `gene_symbol` or `ID`, refering to the barcode name and another column called `spacer` which is the full nucleotide string of the region being ISS'd (e.g. `AATTGGCC`) if fewer nucleotides than the full `spacer` are sequenced only in situ only the relevant bases will be matched. 

Once all variables are set the snakefile can be run by activating the environment than typing
```
snakemake --cores all
```
The script is relatively memory efficient, if you find it crashing you may need to cut the number of cores down.  Alternatively, if you run code some of the modules on a GPU, you may need to cut concurrency to ensure the GPU RAM isn't overwhelmed.

### Registration to Morphological Phenotyping Assays
The script is currently setup to register Ab phenotyping to ISS.

`pheno_dir` should properly point to your phenotyping images.  

### Registration to Xenium Spatial Transcriptomics
Xenium DAPI images can be directly used to register to tissue ISS.

Set `pheno_img` to the location of your DAPI morphology ome.tiff file.  The DAPI Z plane used for registering the data is currently set to `Z=0` on line 50 of `register_xenium.py`.  Change this as needed.

To adapt the script to register Xenium to ISS the following rules should also be removed:
```
stitch_pheno # Xenium is already stitched
segment_phenotype # Xenium is already segmented
bg_subtract_pheno # Not needed
get_nuclei_masks # Xenium is already segmented
```

Additionally the keyword argument `--xenium` should be added to the rule `call_peaks`.
