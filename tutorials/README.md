# Example Usage

## scHiEnhance on Dip-C Tan et al. Cell mousebrain dataset

### Preparation
```
git clone https://github.com/AmyTanJ/scHiEnhance.git
```
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
#### 5. Output from scHiEnhance
Under output directory:

* Path/to/outdir/
   * h5_hic_dataset.h5py: A file includes all patches information.
   * patch_wise_VAE.pth: A file storing patch-wise VAE model.
   * patch_wise_features.hdf5: A file including patch-wise representations and A/B compartment profile.
   * poe_z.hdf5: A file including cell-wise representations.
   * poe_impu_rx.hdf5: A file including reconstructed patch-wise representations from POE framework.
   * POE_VAE.save: A file storing POE model.
   * impu_rec.hdf5: A file storing imputed high-resolution scHi-C contact maps.
#### 6. Visualization
##### 6.1 Clsutering
```
python Visualization.py -dir outdir/ -cell_type cell_label_path/
```
<div align=center><img width="650" height="400" src="https://github.com/AmyTanJ/scHiEnhance/blob/main/figs/10kb_Dip-C_clustering_result.png"/></div>

##### 6.2 Generation
```
python POE_generation.py -dir outdir/ -gen_epoch 5 -gen_num 10
```
Output is a generation_rx_${gen_num}_${gen_epoch}.hdf5 file.
