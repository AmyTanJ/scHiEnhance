import argparse
import os
import random
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader, random_split
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def set_seed(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================

class DataSet(Dataset):
    def __init__(self, h5file, index, num_conditions):
        """
        Args:
            h5file (h5py.File): opened HDF5 file
            index (list): list of sample paths
            num_conditions (int): number of condition dimensions (patch_id length)
        """
        self.dataset = h5file
        self.index = index
        self.num_conditions = num_conditions
        self.preprocessed_labels = self._preprocess_labels()

    def _preprocess_labels(self):
        labels = []
        for idx in self.index:
            sample = self.dataset[idx]
            labels.append([eval(l) for l in sample["label"][:]])
        return labels

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sample = self.dataset[self.index[i]]
        data = torch.from_numpy(sample["img"][:])
        data[data >= 1] = 1
        mask = torch.from_numpy(sample["mask"][:])
        condition = torch.from_numpy(sample["patch_id"][:]).float()
        assert condition.shape[0] == self.num_conditions, \
            f"Expected condition dim {self.num_conditions}, got {condition.shape[0]}"
        label = self.preprocessed_labels[i]
        return data, label, mask, condition

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_conditions, dropout=0.2):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(200 * 200 + num_conditions, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim * 2),
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        h = self.net(x)
        mu = h[:, : self.z_dim]
        sigma = F.softplus(h[:, self.z_dim:]) + 1e-6
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_conditions, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_conditions, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 200 * 200),
            nn.Sigmoid(),
        )

    def forward(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_conditions):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim, num_conditions)
        self.decoder = Decoder(z_dim, hidden_dim, num_conditions)

    def forward(self, x, cond):
        mu, sigma = self.encoder(x, cond)
        z = Normal(mu, sigma).rsample()
        recon = self.decoder(z, cond)
        return recon, mu, sigma

    def loss(self, recon, x, mu, sigma, mask):
        recon_loss = F.binary_cross_entropy(
            recon * mask, x * mask, reduction="sum"
        )
        kl = -0.5 * torch.sum(1 + torch.log(sigma) - mu.pow(2) - sigma)
        return recon_loss + kl, recon_loss, kl

class EarlyStopping:
    def __init__(self, patience):
        """
        Args:
            patience (int): number of epochs with no improvement before stopping
        """
        self.patience = patience
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, loss):
        if self.best is None or loss < self.best:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                
def getZ_list(indices, dir):
    """
    Generate patch-wise representations.
    """
    H5File = h5py.File(dir + '/patch_wise_features.hdf5', 'a')
    cell_nums = sorted(list(np.unique(indices)))
    vae.eval()
    for data, ind, mask, condition in cell_loader:
        data, mask, condition = data.to(device), mask.to(device), condition.to(device)
        mu, std = vae.get_z_params(data, condition)
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
    H5File = h5py.File('impu_rec_mask.hdf5', 'r+')
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional VAE for patch-wise scHi-C data")

    parser.add_argument("--dir", type=str, required=True, help="root directory where high resolution intrachromosomal Hi-C patch-wise profile is stored")
    parser.add_argument("--AB_dir", type=str, required=True, help="root directory where scHi-C A/B compartment profile is stored,
                        dictionary is supported{'cell_1': num1,'cell_2': num2 }")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model and logs")

    parser.add_argument("-n", "--num_epochs", type=int, default=800, help="Number of training epochs")

    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-5, help="Learning rate for Adam optimizer")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader worker processes")

    parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of samples used as validation set")

    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA if available")

    parser.add_argument("--num_conditions", type=int, default=1266, help="Dimension of condition vector (patch_id length), default: 1266")

    parser.add_argument("--z_dim", type=int, default=16, help="Latent dimension of VAE")

    parser.add_argument("--hidden_dim", type=int, default=2000, help="Hidden layer dimension of encoder/decoder")

    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    h5 = h5py.File(args.indir + '/h5_hic_dataset.hdf5', 'a')
    keys = list(h5.keys())

    index = []
    for k in keys:
        for sub in h5[k]:
            index.append(f"{k}/{sub}")
    cellindex = []
    for i in images:
        cellindex.append(i.split('/')[0])
    cellindex = np.unique(cellindex)
    dataset = DataSet(h5, index, args.num_conditions)
    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoaderX(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoaderX(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = VAE(z_dim=args.z_dim, hidden_dim=args.hidden_dim, num_conditions=args.num_conditions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stop = EarlyStopping(args.patience)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0

        for x, _, mask, cond in train_loader:
            x, mask, cond = x.to(device), mask.to(device), cond.to(device)
            optimizer.zero_grad()
            recon, mu, sigma = model(x, cond)
            loss, _, _ = model.loss(recon, x, mu, sigma, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _, mask, cond in test_loader:
                x, mask, cond = x.to(device), mask.to(device), cond.to(device)
                recon, mu, sigma = model(x, cond)
                loss, _, _ = model.loss(recon, x, mu, sigma, mask)
                test_loss += loss.item()

        test_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch:03d} | "f"Train Loss: {train_loss:.4f} | "f"Val Loss: {test_loss:.4f}")

        early_stop(test_loss)
        if early_stop.stop:
            print("Early stopping triggered.")
            break

    torch.save(model, os.path.join(args.output_dir, "patch_wise_VAE.pth"))
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
    args = parse_args()
    main(args)
