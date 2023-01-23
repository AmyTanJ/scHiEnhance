import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn as nn
import numpy as np
from torch.distributions.utils import broadcast_all
from torch.nn.functional import binary_cross_entropy_with_logits
import skimage.transform as st
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import hicstraw
import scanpy as sc
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from prefetch_generator import BackgroundGenerator
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

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
    def __init__(self, z_dim, hidden_dim, input_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.BN1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.relu(self.BN1(self.fc1(x)))
        z_loc = self.fc21(x)
        z_scale = torch.exp(self.fc22(x))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # setup the non-linearities
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
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
        if use_cuda:
            x = x.cuda()
            m = m.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x, m)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=True):
    # initialize loss accumulator
    test_loss = 0.
    # do a testing epoch over each mini-batch x returned
    # by the data loader
    for x, index, m in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            m = m.cuda()
        test_loss += svi.evaluate_loss(x, m)

    # return testing loss
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

class VAE(nn.Module):
    def __init__(self, z_dim=16, hidden_dim=2000, use_cuda=True, patch_size = 200):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, patch_size*patch_size)
        self.decoder = Decoder(z_dim, hidden_dim, patch_size*patch_size)
        
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
        self.path = path + ’patch_wise_VAE.save‘
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

def getZ_list(indices, dir):
    """
    Generate patch-wise representations.
    """
    H5File = h5py.File(dir + '/patch_wise_features.hdf5', 'a')
    cell_nums = sorted(list(np.unique(indices)))
    for x, ind, m in cell_loader:
        x = x.cuda()
        mu, std = vae.getZ(x)
        mu=mu.cpu().detach().numpy()
        std=std.cpu().detach().numpy()
        z = np.concatenate((mu, std),axis=1).tolist()
        for index, i in enumerate(ind[0]):
            if i in H5File.keys():
                H5File[i].resize((H5File[i].shape[0] + np.array(z[index]).shape[0]), axis=0)
                H5File[i][-np.array(z[index]).shape[0]:] = z[index]
            else:
                H5File.create_dataset(i, data = z[index], compression="gzip", maxshape=(None,))
    H5File.close()
    return None

def vis(dir):
    """
    prepare H5File['sum'] for POE input.
    """
    count = 0
    index = []
    H5File = h5py.File(dir + '/patch_wise_features.hdf5', 'r+')
    # del H5File['sum']
    for key in H5File.keys():
        index.append(H5File[key].name)
    for key in H5File.keys():
        if count == 0:
            H5File.create_dataset('sum', data = H5File[key][:], compression="gzip", maxshape=(None,))
            count+= 1
        else:
            H5File['sum'].resize((H5File['sum'].shape[0] + H5File[key][:].shape[0]), axis=0)
            H5File['sum'][-H5File[key][:].shape[0]:] = H5File[key][:]
    return None

def load_obj(obj, name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main(args):
    H5Dataset = h5py.File(args.indir + '/h5_hic_dataset.hdf5', 'a')

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
    test_size = int(args.test_ratio*len(images))
    train_size = len(images) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(celldata, [train_size, test_size],generator=torch.Generator().manual_seed(0))

    train_loader = DataLoaderX(train_dataset, args.batch_size, args.num_workers)
    test_loader = DataLoaderX(test_dataset, args.batch_size, args.num_workers)
    cell_loader=DataLoaderX(celldata, args.batch_size, args.num_workers)

    # # clear param store
    pyro.clear_param_store()
    # # setup the VAE
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

    getZ_list(cellindex, args.dir)
    vis(args.dir)

    n_AB_compart = np.load(args.AB_dir + 'scAB_compartment.npy', allow_pickle='TRUE').item()
    H5File = h5py.File(args.dir + '/patch_wise_features.hdf5', 'a')

    for key in tqdm(H5Dataset.keys()):
        for i in n_AB_compart.keys():
            if key == i:
                H5Dataset[key].resize((H5Dataset[key].shape[0] + n_AB_compart[i][:].shape[0]), axis=0)
                H5Dataset[key][-n_AB_compart[i][:].shape[0]:] = n_AB_compart[i][:]
    H5Dataset.close()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where high resolution intrachromosomal Hi-C patch-wise profile is stored"
    )
    parser.add_argument(
        "--AB_dir", type=str, required=True,
        help="root directory where scHi-C A/B compartment profile is stored,
        dictionary is supported{'cell_1': num1,'cell_2': num2 }"
    )
    parser.add_argument(
        "-n", "--num_epochs", default=200, type=int, help="number of training epochs"
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
        "--batch_size", default=5200, type=int, help="batch size"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of workers"
    )
    parser.add_argument(
        "--patience", default=20, type=int, help="early-stopping patience"
    )
    args = parser.parse_args()

    mian(args)
