import argparse
import os
import torch
from torch.utils.data import Dataset,DataLoader, IterableDataset
import torch.nn as nn
import numpy as np
from itertools import cycle, islice
import skimage.transform as st
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import hicstraw
import scanpy as sc
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import random

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
        data= torch.from_numpy(np.float32(sample[:]))
        label = self.dataset_num[idx]
        return data, label

class AB_Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, AB_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(AB_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        self.BN2 = nn.BatchNorm1d(z_dim*2)
        # self.softplus = nn.Softplus()

    def forward(self, x):
        hidden = self.relu(self.BN1(self.fc1(x)))
        z_loc = self.relu(self.BN2(self.fc2(hidden)))
        return z_loc

class Ten_kb_Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, hr_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(hr_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        z_loc = self.relu(self.fc2(x))
        return z_loc

class AB_Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, AB_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, AB_dim)
        # setup the non-linearities
        self.BN = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.relu(self.BN(self.fc1(z)))
        loc_img = self.Sigmoid(self.fc21(hidden))
        return loc_img

class Ten_kb_Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, hr_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hr_dim)
        # setup the non-linearities
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.Softplus()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        x = self.relu(self.fc1(z))
        loc_img = self.Sigmoid(self.fc2(x))
        return loc_img

class Encoder(nn.Module):
    def __init__(self, z_dim = 32, hidden_dim_1=4000, hidden_dim_2=1000, AB_dim, hr_dim):
        super().__init__()
        self.Ten_encoder = Ten_kb_Encoder(z_dim*2, hidden_dim_1, hr_dim)
        self.AB_encoder = AB_Encoder(z_dim*2, hidden_dim_2, AB_dim)
        self.combined_fc = nn.Conv2d(2, 1, kernel_size=1, stride=1)
        self.fc_mu = nn.Linear(z_dim*2, z_dim)
        self.fc_var = nn.Linear(z_dim*2, z_dim)

    def forward(self, x):
        encoded_tenkb = self.Ten_encoder(x[:, :hr_dim])
        encoded_tenkb = torch.unsqueeze(encoded_tenkb, 1)
        encoded_ab = self.AB_encoder(x[:, hr_dim:])
        encoded_ab = torch.unsqueeze(encoded_ab, 1)
        combined = torch.cat(
            [encoded_tenkb, encoded_ab], dim=1
        )
        combined = torch.unsqueeze(combined, 3)
        result = self.combined_fc(combined)
        result = torch .flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z_loc = self.fc_mu(result)
        z_scale = torch.exp(self.fc_var(result))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, z_dim = 32, hidden_dim_1=4000, hidden_dim_2=1000, AB_dim, hr_dim):
        super().__init__()
        self.Ten_decoder = Ten_kb_Decoder(z_dim, hidden_dim_1, hr_dim)
        self.AB_decoder = AB_Decoder(z_dim, hidden_dim_2, AB_dim)

    def forward(self, z):
        recon_tenkb = self.Ten_decoder(z)
        recon_ab = self.AB_decoder(z)
        recon_combined = torch.cat(
            [recon_tenkb, recon_ab], dim=1
        )
        return recon_combined

def train(svi, train_loader, use_cuda=True, getz = False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, index in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=True):
    # initialize loss accumulator
    test_loss = 0.
    # do a testing epoch over each mini-batch x returned
    # by the data loader
    for x, index in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        test_loss += svi.evaluate_loss(x)

    # return testing loss
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

class VAE(nn.Module):
    def __init__(self, z_dim_1=32, z_dim_2=32, hidden_dim_1=4000, hidden_dim_2=1000, use_cuda=True, cell_size = 80104, AB_dim = 2376, hr_dim = 32*2429):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim_1, hidden_dim_1, hidden_dim_2, AB_dim, hr_dim)
        self.decoder = Decoder(z_dim_2, hidden_dim_1, hidden_dim_2, AB_dim, hr_dim)
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim_2
        self.cell_size = cell_size

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the combined latent code z
            loc_img = self.decoder(z)
            loc_img = loc_img.reshape(-1, self.cell_size)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.cell_size))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
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

def main(args):
    H5Dataset = h5py.File(args.dir + '/patch_wise_features.hdf5', 'a')
    cell_keys =[]
    for key in H5Dataset.keys():
        if key == 'sum':
            continue
        cell_keys.append(key)
    
    celldata=DataSet(H5Dataset, cell_keys)
    cell_loader=DataLoaderX(celldata, args.batch_size, args.num_workers)
    
    # clear param store
    pyro.clear_param_store()
    # setup the VAE
    vae = torch.load(args.dir + '/POE_VAE.save')
    
    # setup the optimizer
    random.seed(0)

    for n in tqdm(range(args.gen_epoch)):
        H5File = h5py.File(args.dir + ('/generation_rx_{0}_{1}.hdf5').format(args.gen_num, n), 'a')
        for x, ind in cell_loader:
            x = x.cuda()
            z_loc, z_scale = vae.encoder(x)
            # sample in latent space
            for j in range(args.gen_num):
                z = dist.Normal(z_loc, z_scale).sample()
                loc_img = vae.decoder(z)
                loc_img = loc_img.cpu().detach().numpy()
                for num, i in enumerate(ind):
                    group_identity = H5File.create_group('{0}_{1}'.format(i, j))
                    group_identity.create_dataset('rx', data = loc_img[num], compression="gzip", maxshape=(None,))
                    group_identity.create_dataset('z', data = (z[num]).cpu().detach().numpy(), compression="gzip", maxshape=(None,))       
        H5File.close()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where patch-wise representations are stored"
    )
    parser.add_argument(
        "--gen_epoch", type=int, required=True,
        help="Number of generation"
    )
    parser.add_argument(
        "--gen_num", type=int, required=True,
        help="The multiple of generative cells over raw cells"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="whether to use cuda"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of workers"
    )
    args = parser.parse_args()

    mian(args)
