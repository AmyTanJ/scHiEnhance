import argparse
import os
import random
import time
import datetime
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataSet(Dataset):
    def __init__(self, h5file, keys):
        """
        Args:
            h5file (h5py.File): opened HDF5 file
            keys (list): dataset keys (cell ids)
        """
        self.dataset = h5file
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = torch.from_numpy(
            np.asarray(self.dataset[key], dtype=np.float32)
        )
        return data, key

class AB_Encoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(z_dim)

    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        z = self.relu(self.bn2(self.fc2(h)))
        return z

class TenKb_Encoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        z = self.relu(self.fc2(x))
        return z

class AB_Decoder(nn.Module):
    def __init__(self, out_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.bn(self.fc1(z)))
        return self.sigmoid(self.fc2(h))

class TenKb_Decoder(nn.Module):
    def __init__(self, out_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        return self.sigmoid(self.fc2(h))

class Encoder(nn.Module):
    def __init__(self, tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.tenkb_enc = TenKb_Encoder(tenkb_dim, z_dim * 2, hidden_dim_1)
        self.ab_enc = AB_Encoder(ab_dim, z_dim * 2, hidden_dim_2)

        self.combine = nn.Conv2d(2, 1, kernel_size=1)
        self.fc_mu = nn.Linear(z_dim * 2, z_dim)
        self.fc_var = nn.Linear(z_dim * 2, z_dim)

    def forward(self, x):
        tenkb = self.tenkb_enc(x[:, :self.tenkb_enc.fc1.in_features])
        ab = self.ab_enc(x[:, self.tenkb_enc.fc1.in_features:])

        tenkb = tenkb.unsqueeze(1)
        ab = ab.unsqueeze(1)

        combined = torch.cat([tenkb, ab], dim=1).unsqueeze(3)
        h = self.combine(combined).flatten(start_dim=1)

        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_var(h)) + 1e-8
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.tenkb_dec = TenKb_Decoder(tenkb_dim, z_dim, hidden_dim_1)
        self.ab_dec = AB_Decoder(ab_dim, z_dim, hidden_dim_2)

    def forward(self, z):
        tenkb = self.tenkb_dec(z)
        ab = self.ab_dec(z)
        return torch.cat([tenkb, ab], dim=1)

class VAE(nn.Module):
    def __init__(self, tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.encoder = Encoder(
            tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2
        )
        self.decoder = Decoder(
            tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2
        )

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = Normal(mu, sigma).rsample()
        recon = self.decoder(z)
        return recon, mu, sigma

    def getZ(self, x):
        return self.encoder(x)
        
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
                
def elbo_loss(recon, x, mu, sigma):
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + torch.log(sigma) - mu.pow(2) - sigma)
    return recon_loss + kl

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

def getRX_list(dir):
    """
    Generate RX to reconstruct completed scHi-C contact maps.
    """
    H5File = h5py.File(dir + '/poe_impu_rx.hdf5', 'a')
    vae.eval()
    with torch.no_grad():
        for x, ind in cell_loader:
            x = x.cuda()
            reconstructed_x, _, _ = vae(x)
            rx = reconstructed_x.cpu().numpy()
            for index, i in enumerate(ind):
                H5File.create_dataset(i, data=rx[index], compression="gzip", maxshape=(None,))
    H5File.close()
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="POE VAE for 10kb Hi-C + A/B compartment features"
    )

    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where patch-wise representations are stored"
    )

    parser.add_argument("--tenkb_dim", type=int, default=40512,
                        help="Dimension of 10kb input feature")
    parser.add_argument("--ab_dim", type=int, default=2684,
                        help="Dimension of A/B compartment feature")

    parser.add_argument("--z_dim", type=int, default=32,
                        help="Latent dimension")
    parser.add_argument("--hidden_dim_1", type=int, default=3500,
                        help="Hidden dimension for 10kb encoder/decoder")
    parser.add_argument("--hidden_dim_2", type=int, default=512,
                        help="Hidden dimension for AB encoder/decoder")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=400,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--test_ratio", type=float, default=0.3,
                        help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping patience")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA if available")

    return parser.parse_args()


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    H5Dataset = h5py.File(args.dir + '/patch_wise_features.hdf5', 'a')
    keys = [k for k in h5.keys() if k != "sum"]

    dataset = DataSet(h5, keys)
    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size

    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoaderX(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoaderX(
        test_ds, batch_size=args.batch_size
    )

    vae = VAE(
        args.tenkb_dim,
        args.ab_dim,
        args.z_dim,
        args.hidden_dim_1,
        args.hidden_dim_2,
    ).to(device)

    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.learning_rate
    )

    for epoch in range(args.num_epochs):
        vae.train()
        train_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, sigma = vae(x)
            loss = elbo_loss(recon, x, mu, sigma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon, mu, sigma = vae(x)
                val_loss += elbo_loss(recon, x, mu, sigma).item()

        val_loss /= len(test_loader.dataset)

        print(f"[Epoch {epoch:03d}] " f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        early_stop(test_loss)
        if early_stop.stop:
            print("Early stopping triggered.")
            break

    torch.save(vae,cell_wise_model.pth")
    
    # For clustering combined latent space
    getZ_list(args.dir)
    
    # The input for patch-wise decoder
    getRX_list(args.dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)

