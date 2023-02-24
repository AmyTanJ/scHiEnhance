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
#### 5. Visualization
##### 5.1 Clsutering
```
python Visualization.py -dir outdir/ -cell_type cell_label_path/
```
<div align=center><img width="650" height="400" src="https://github.com/AmyTanJ/scHiEnhance/blob/main/figs/10kb_Dip-C_clustering_result.png"/></div>

##### 5.2 Generation
```
python POE_generation.py -dir outdir/ -gen_epoch 5 -gen_num 10
```
