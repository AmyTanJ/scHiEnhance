# Example Usage

## scHiEnhance on Dip-C Tan et al. Cell mousebrain dataset

### Preparation
#### Download input files
Download the Dip-C data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162511

### Running scHiEnhance
#### 1. Data processing
```
cd scHiEnhance
python Processor.py -indir indir/ -outdir outdir/
```
#### 2. Train the patch-wise VAE
```
python Patch_wise_VAE.py -dir outdir/ -AB_dir AB_dir/
```
#### 3. Train the POE framework
```
python POE_VAE.py -dir outdir/
```
#### 4. Reconstructing imputed high-resolution scHi-C contact maps
```
python Reconstruction.py -dir outdir/
```
