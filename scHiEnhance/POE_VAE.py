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
        
def getRX_list(dir):
    """
    Generate RX to reconstruct completed scHi-C contact maps.
    """
    H5File = h5py.File(dir + '/poe_impu_rx.hdf5', 'a')
    for x, ind in cell_loader:
        x = x.cuda()
        rx = vae.reconstruct_img(x)
        rx = rx.cpu().detach().numpy()
        for ind, i in enumerate(ind):
            H5File.create_dataset(i, data = rx[ind], compression="gzip", maxshape=(None,))
    H5File.close()
    return None

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='.', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path + '/POE_VAE.save'
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss

def getZ_list(dir):
    """
    Generate combined latent code z for clustering.
    """
    H5File = h5py.File(dir + '/poe_z.hdf5', 'a')
    for x, ind in cell_loader:
        x = x.cuda()
        mu, std = vae.getZ(x)
        mu=mu.cpu().detach().numpy()
        std=std.cpu().detach().numpy()
        z = mu.tolist()
        for ind, i in enumerate(ind):
            if i in H5File.keys():
                H5File[i].resize((H5File[i].shape[0] + np.array(z[ind]).shape[0]), axis=0)
                H5File[i][-np.array(z[ind]).shape[0]:] = z[ind]
            else:
                H5File.create_dataset(i, data = z[ind], compression="gzip", maxshape=(None,))
    H5File.close()
    return None
    
def main(args):
    H5Dataset = h5py.File(args.dir + '/patch_wise_features.hdf5', 'a')
    cell_keys =[]
    for key in H5Dataset.keys():
        if key == 'sum':
            continue
        cell_keys.append(key)
    
    celldata=DataSet(H5Dataset, cell_keys)
    test_size = int(args.test_ratio*len(cell_keys))
    train_size = len(cell_keys) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(celldata, [train_size, test_size],generator=torch.Generator().manual_seed(0))
    
    train_loader = DataLoaderX(train_dataset, args.batch_size, args.num_workers)
    test_loader = DataLoaderX(test_dataset, args.batch_size, args.num_workers)
    cell_loader=DataLoaderX(celldata, args.batch_size, args.num_workers)
    
    # clear param store
    pyro.clear_param_store()
    # setup the VAE
    vae = VAE(use_cuda=args.cuda)
    
    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)
    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
    train_elbo = []
    test_elbo = []
    early_stopping = EarlyStopping(args.patience, verbose=True, path = args.dir)
    
    # training loop
    for epoch in tqdm(range(args.num_epochs)):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=args.cuda)
        train_elbo.append(-total_epoch_loss_train)
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=args.cuda)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d]  training loss: %.4f" % (epoch, total_epoch_loss_train))
        print("[epoch %03d]  testing loss: %.4f" % (epoch, total_epoch_loss_test))
        early_stopping(total_epoch_loss_test, vae)
        if early_stopping.early_stop:
            print("Early stopping")
            getZ_list()
            vis()
            plt.plot(range(1,len(train_elbo)+1),-np.asarray(train_elbo), label='Training Loss')
            plt.plot(range(1,len(test_elbo)+1),-np.asarray(test_elbo),label='Validation Loss')
            # find position of lowest validation loss
            minposs = test_elbo.index(-min(-np.asarray(test_elbo)))+1 
            plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
            plt.title('VAE train elbo')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.xlim(0, len(train_elbo)+1) # consistent scale
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            break

    # For clustering combined latent space
    getZ_list(args.dir)
    # The input for patch-wise decoder
    getRX_list(args.dir)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where patch-wise representations are stored"
    )
    parser.add_argument(
        "-n", "--num_epochs", default=800, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=1.0e-5, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="whether to use cuda"
    )
    parser.add_argument(
        "--test_ratio", default=0.3, type=float, help="test dataset frequency"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of workers"
    )
    parser.add_argument(
        "--patience", default=100, type=int, help="early-stopping patience"
    )
    args = parser.parse_args()

    mian(args)
