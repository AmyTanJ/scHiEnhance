import argparse
import os
import torch
from torch.utils.data import Dataset,DataLoader, IterableDataset
import torch.nn as nn
import numpy as np
from itertools import cycle, islice
from torch.distributions.utils import broadcast_all
from torch.nn.functional import binary_cross_entropy_with_logits
import skimage.transform as st
import h5py
import matplotlib.pyplot as plt
import hicstraw
from tqdm import tqdm
import scanpy as sc
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class DataSet(Dataset):
    def __init__(self,fname, index):
        self.dataset = fname
        self.dataset_num = index
    
    def __len__(self):
        return len(self.dataset_num)
    
    def __getitem__(self,idx):
        sample = self.dataset[self.dataset_num[idx]]
        label = []
        for i in sample['label'][:]:
            label.append(eval(i))
        data = torch.from_numpy(sample['img'][:])
        mask = torch.from_numpy(sample['mask'][:])
        return data, label, mask

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(200*200, 2000)
        self.fc21 = nn.Linear(2000, z_dim)
        self.fc22 = nn.Linear(2000, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.BN1 = nn.BatchNorm1d(2000)

    def forward(self, x):
        x = self.relu(self.BN1(self.fc1(x)))
        z_loc = self.fc21(x)
        z_scale = torch.exp(self.fc22(x))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim,2000)
        self.fc2 = nn.Linear(2000, 200*200)
        # setup the non-linearities
        # self.softplus = nn.Softplus()
        self.BN1 = nn.BatchNorm1d(2000)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        x = self.relu(self.BN1(self.fc1(z)))
        loc_img = self.Sigmoid(self.fc2(x))
        return loc_img

def train(svi, train_loader, use_cuda=True, getz = False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, index, m in train_loader:
        # if on GPU put mini-batch into CUDA memory
        # print("x")
        if use_cuda:
            x = x.cuda()
            m = m.cuda()
        # do ELBO gradient and accumulate loss
        # print("x.cuda")
        epoch_loss += svi.step(x, m)
#         print(epoch_loss)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=True):
    # initialize loss accumulator
    test_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, index, m in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            m = m.cuda()
        # do ELBO gradient and accumulate loss
        test_loss += svi.evaluate_loss(x, m)

    # return epoch loss
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

class VAE(nn.Module):
    def __init__(self, z_dim=16, hidden_dim=2000, use_cuda=True, patch_size = 200):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.patch_size = patch_size

    # define the model p(x|z)p(z)
    def model(self, x, m):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            loc_img = loc_img.reshape(-1,self.patch_size*self.patch_size)
            pyro.sample("obs", dist.Bernoulli(loc_img).mask(m).to_event(1), obs=x.reshape(-1, self.patch_size*self.patch_size))
            
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, m):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
    
    def getZ(self, x):
        # enczode image x
        z_loc, z_scale = self.encoder(x)
        return (z_loc,z_scale)
    
    def getRZ(self, z):
        loc_img = self.decoder(z)
        return loc_img

def getRX_list(key_sum, size_list, images, chr_num = 19):
    """
    Reconstruct imputed scHi-C contact maps.
    """
    n_AB_compart = np.load(args.AB_dir + 'scAB_compartment.npy', allow_pickle='TRUE').item()
    c=list(n_AB_compart)
    dim_AB_compart = n_AB_compart[c[0]].shape[1]
    for key in tqdm(key_sum):
        H5File = h5py.File(args.dir + '/poe_impu_rx.hdf5')
        rx = H5File[key][:-dim_AB_compart]
        H5File.close()
        rx = rx.reshape(2429, 32)
        rx = torch.from_numpy(rx[:,:16])
        rxrx = vae.getRZ(rx.cuda())
        rxrx = rxrx.cpu().detach().numpy()
        for chro in range(1,chr_num+1):
            H5Dataset = h5py.File(args.dir+'/h5_hic_dataset.hdf5')
            rec_images = []
            for ims in images:
                if ims.split("/")[1] == str(chro):
                    i, j = (ims.split("/")[2]).split("_")[0], (ims.split("/")[2]).split("_")[1]
                    rec_images.append((eval(i), eval(j)))
            width = eval(H5Dataset[key_sample][str(chro)]['0_0']['label'][:][2])
            H5Dataset.close()
            full_mat_rx = np.zeros((width, width))
            test_num = size_list[chro-1]
            test_mat = rxrx[test_num[0]:test_num[1]]
            for p in range(len(test_mat)):
                iw, jw = rec_images[p][0], rec_images[p][1]
                sub_rx = test_mat[p].reshape(200, 200)
                h, w = sub_rx.shape
                if iw ==jw:
                    full_mat_rx[iw:iw + h, jw:jw + w]+= sub_rx
                else:
                    full_mat_rx[iw:iw + h, jw:jw + w]+= sub_rx
                    full_mat_rx[jw:jw + w, iw:iw + h] += sub_rx.T
            mat_up = full_mat_rx-np.tril(full_mat_rx,-201)-np.triu(full_mat_rx,201)
            H5F = h5py.File(args.dir + '/impu_rec.hdf5','a')
            if key in H5F.keys():
                H5F[key].create_dataset('{0}'.format(chro), data = mat_up, compression="gzip")
            else:
                group = H5F.create_group(key)
                group.create_dataset('{0}'.format(chro), data = mat_up, compression="gzip")
            H5File.close()
    return None

def main(args):
    H5Dataset = h5py.File(args.dir + '/h5_hic_dataset.hdf5')

    im = []
    def get_data_items(name, obj):
        if len(name.split('/')) == 3:
            if name.split('/')[2] == 'img':
                # print((name.split('/')[1]).split('_')[0], (name.split('/')[1]).split('_')[1])
                if int((name.split('/')[1]).split('_')[0])>= int((name.split('/')[1]).split('_')[1]):
                    im.append('/'.join(name.split('/')[:2]))

    for key in H5Dataset.keys():
         key_sample = key
         break

    H5Dataset[key_sample].visititems(get_data_items)

    images = []
    cell_images = []
    for key in H5Dataset.keys():
        cell_images.append(key)
        for i_m in im:
            cell_im = '/'.join([key, i_m])
            images.append(cell_im)

    cellindex = []
    for i in images:
        cellindex.append(i.split('/')[0])
    cellindex = np.unique(cellindex)

    celldata=DataSet(H5Dataset, images)
    cell_loader=DataLoaderX(celldata, batch_size=6240, num_workers=4)

    # clear param store
    pyro.clear_param_store()
    vae = torch.load(args.dir + ’/patch_wise_VAE.save‘')
    H5File = h5py.File(args.dir + '/poe_impu_rx.hdf5')
    key_sum = []
    for key in H5File.keys():
        key_sum.append(key)

    H5File.close()
    H5Dataset.close()

    size_list = [(0,193),(193,374),(374,533),(533,688),(688,837),(837,984),(984,1127),(1127,1254),(1254,1377),
     (1377,1506),(1506,1627),(1627,1746),(1746,1865),(1865,1988),(1988,2091),(2091,2188),(2188,2281),
     (2281,2370),(2370,2429)]

    getRX_list(key_sum, size_list,rec_images)



if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where both high resolution intrachromosomal Hi-C interaction matrices are stored"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="whether to use cuda"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of workers"
    )
    args = parser.parse_args()

    mian(args)
