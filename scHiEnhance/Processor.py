import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import h5py
import hicstraw
from tqdm import tqdm

def carn_divider(hic, cell_num, chr_num, gi, chunk=200, stride=200, bound=201, padding=True):
    divide(hic, cell_num, chr_num, gi, chunk, stride, bound, padding)

def divide(mat, cell_num, chr_num, gi, chunk_size=200, stride=200, bound=201, padding=True, verbose=False):
    """
    Dividing method.
    """
    result = []
    size = mat.shape[0]
    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i - j) <= bound and i + chunk_size < height and j + chunk_size < width and i>=j:
                subImage = mat[i:i + chunk_size, j:j + chunk_size]
                subImage = subImage.astype(np.float32)
                subImage=torch.from_numpy(subImage)
                subImage = subImage.reshape(200*200)
                sample = gi.create_group('{0}_{1}'.format(i, j))
                sample.create_dataset('img', data = subImage, compression="gzip")
                strList = [cell_num, chr_num, size, i, j]
                asciiList = [repr(n).encode("ascii", "ignore") for n in strList]
                sample.create_dataset('label', data = asciiList)
    if verbose: print(
        f'[Chr{chr_str}] Deviding HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, '
        f'stride={stride}, bound={bound}')

def res_change(length, res):
    """
    Change resolution of single cell .hic file.
    """
    if length % res == 0:
        return length // res
    else:
        return length // res + 1

def processor(args):
    """
    Divde scHi-C contact map into square patches.
    """
    path = args.indir
    save_path = args.outdir + '/h5_hic_dataset.hdf5'
    resolution = args.resolution
    cell_list = []
    for filename in os.listdir(path):
        cell_list.append(filename)
    cell_path_list = [path + '/' +  i for i in cell_list]
    H5File = h5py.File(save_path, 'a')
    for j in tqdm(range(len(cell_path_list))):
        hic = hicstraw.HiCFile(cell_path_list[j])
        filename = f'{(cell_path_list[j].split("/")[-1]).split(".")[0]}'
        cell_identity = H5File.create_group(filename)
        for ind, ch in enumerate(args.chr_list):
            for i in range(len(hic.getChromosomes())):
                if ch == hic.getChromosomes()[i].name:
                    size = res_change(hic.getChromosomes()[i].length, resolution)
            gen = hicstraw.straw('observed', 'NONE', cell_path_list[j], ch,ch,'BP', resolution)
            A = np.zeros((size, size))
            for i in range(len(gen)):
                p1 = gen[i].binX // resolution
                p2 = gen[i].binY // resolution
                val = gen[i].counts
                A[p1, p2] += val
                if p1 != p2:
                    A[p2, p1] += val
            A = A.astype(np.float32)
            group_identity = cell_identity.create_group('{0}'.format(ch))
            carn_divider(A, (cell_path_list[j].split("/")[-1]).split(".")[0], ch, group_identity, args.chunk, args.stride, args.bound, args.padding)
    H5File.close()
return None

def masking(args):
    """
    Generate maksing matrics.
    """
    masking_ratio = args.masking_ratio
    save_path = args.outdir+'/h5_hic_dataset.hdf5'
    H5File = h5py.File(save_path, 'r+')
    im = []
    def get_data_items(name, obj):
        if len(name.split('/')) == 3:
            if name.split('/')[2] == 'img':
                if int((name.split('/')[1]).split('_')[0])>= int((name.split('/')[1]).split('_')[1]):
                    im.append('/'.join(name.split('/')[:2]))

   for key in H5Dataset.keys():
        key_sample = key
        break

    H5Dataset[key_sample].visititems(get_data_items)

    images = []
    cell_images = []
    for key in H5File.keys():
        cell_images.append(key)
        for i_m in im:
            cell_im = '/'.join([key, i_m])
            images.append(cell_im)

    np.random.seed(0)
    for key in tqdm(images):
        if eval(H5File[key]['label'][3]) == eval(H5File[key]['label'][4]):
            mask = np.where(np.tril(H5File[key]['img'][:].reshape(200, 200)) <= 0, 0, 1)
            mask_tril = mask[np.tril_indices_from(mask)]
            num_patches = np.sum(mask_tril == 0)
            num_mask = int((1-masking_ratio) * num_patches)
            mask_ind = np.hstack([np.zeros(num_patches - num_mask), np.ones(num_mask)])
            np.random.shuffle(mask_ind)
            mask_tril[mask_tril == 0] = mask_ind
            mask[np.tril_indices_from(mask)] = mask_tril
            mask = mask + mask.T -np.diag(np.diag(mask))
            H5File[key].create_dataset('mask', data = mask.reshape(200*200), compression="gzip")
        else:
            mask = np.where(H5File[key]['img'][:] <= 0, 0, 1)
            num_patches = np.sum(mask == 0)
            num_mask = int((1-masking_ratio) * num_patches)
            mask_ind = np.hstack([np.zeros(num_patches - num_mask), np.ones(num_mask)])
            np.random.shuffle(mask_ind)
            mask[mask == 0] = mask_ind
            H5File[key].create_dataset('mask', data = mask, compression="gzip")
    H5File.close()
return None

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir", type=str, required=True,
        help="root directory where high resolution intrachromosomal Hi-C interaction .hic profile are stored"
    )
    parser.add_argument(
        '--output', type=str, required=True,
            help="output directory to write preprocessing results to"
    )
    parser.add_argument(
        "--chr_list", default=['1','2','3','4','5','6','7','8','9','10',
        '11','12','13','14','15','16','17','18','19'], type=str, nargs='+', help="chromosomes list to analyse"
    )
    parser.add_argument(
        "--resolution", type=int, default=10000, help="single cell Hi-C resolution"
    )
    parser.add_argument(
        "--chunk", type=int, default=200, help="chunk size to split scHi-C"
    )
    parser.add_argument(
        "--stride", type=int, default=200, help="stride size to split scHi-C"
    )
    parser.add_argument(
        "--bound", type=int, default=201, help="bound size to split scHi-C"
    )
    parser.add_argument(
        "--padding", action="store_true", default=True, help="whether padding when dividing scHi-C"
    )
    parser.add_argument(
        "--masking_ratio", type=float, default=0.6, help="masking ratio"
    )
    args = parser.parse_args()

    processor(args)
    masking(args)
