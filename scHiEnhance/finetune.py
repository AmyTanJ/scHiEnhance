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
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm


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
    def __init__(self, h5file, index):
        self.dataset = h5file
        self.index = index
        self.labels = self._preprocess_labels()

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
        return data, mask, condition


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_conditions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200 * 200 + num_conditions, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, z_dim * 2),
        )
        self.z_dim = z_dim

    def forward(self, x, cond):
        h = self.net(torch.cat([x, cond], dim=1))
        mu = h[:, :self.z_dim]
        sigma = F.softplus(h[:, self.z_dim:]) + 1e-6
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_conditions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_conditions, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 200 * 200),
            nn.Sigmoid(),
        )

    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], dim=1))


class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_conditions):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim, num_conditions)
        self.decoder = Decoder(z_dim, hidden_dim, num_conditions)

def fine_tune_loss(rxrx, x, mask):
    return F.binary_cross_entropy(rxrx * mask, x * mask, reduction="sum")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step4: decoder fine-tuning with POE latent samples"
    )
    parser.add_argument("--dir", type=str, required=True,
                        help="root directory where both high resolution intrachromosomal Hi-C interaction matrices are stored")
    parser.add_argument("--AB_dir", type=str, required=True, help="root directory where scHi-C A/B compartment profile is stored," \
                        "dictionary is supported{'cell_1': num1,'cell_2': num2 }")

    parser.add_argument("--num_conditions", type=int, default=1266,
                        help="Number of condition dimensions (patch_id)")
    parser.add_argument("--z_dim", type=int, default=16,
                        help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=2000,
                        help="Decoder hidden dimension")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for decoder fine-tuning")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA if available")

    return parser.parse_args()


def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    # ---------- Load model ----------
    vae = torch.load(args.dir + '/patch_wise_VAE.pth', map_location=device)
    vae.decoder.train()
    optimizer = torch.optim.Adam(
        vae.decoder.parameters(), lr=args.lr
    )

    # ---------- Load AB info ----------
    ab = np.load(args.AB_dir + 'scAB_compartment.npy')
    dim_ab = ab.shape[1]

    # ---------- Load POE rx ----------
    rx_h5 = h5py.File(args.dir + '/poe_impu_rx.hdf5', "r")
    keys = list(rx_h5.keys())

    # ---------- Load input ----------
    input_h5 = h5py.File(args.dir+'/h5_hic_dataset.hdf5', "r")

    # ---------- Preload patch_id ----------
    example_key = list(input_h5.keys())[0]
    patch_ids = []
    for sub in input_h5[example_key]:
        patch_ids.append(input_h5[example_key][sub]["patch_id"][:])
    patch_ids = torch.from_numpy(
        np.asarray(patch_ids).reshape(args.num_conditions, args.num_conditions)
    ).float().to(device)

    # ---------- Training ----------
    for epoch in range(args.epochs):
        total_loss = 0.0

        for key in tqdm(keys, leave=False):
            rx = rx_h5[key][:-dim_ab]
            rx = rx.reshape(-1, args.z_dim * 2)

            mu = torch.from_numpy(rx[:, :args.z_dim]).float().to(device)
            sigma = torch.from_numpy(rx[:, args.z_dim:]).float().to(device)

            z = Normal(mu, sigma).rsample()
            rxrx = vae.decoder(z, patch_ids)

            x = torch.from_numpy(
                input_h5[key]["img"][:]
            ).float().to(device)

            mask = torch.from_numpy(
                input_h5[key]["mask"][:]
            ).float().to(device)

            loss = fine_tune_loss(rxrx, x, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch + 1:03d}] Total loss: {total_loss:.4f}")

    rx_h5.close()
    input_h5.close()
    torch.save(vae, args.dir + 'finetuned_patch_wise_VAE.pth')


if __name__ == "__main__":
    args = parse_args()
    main(args)
