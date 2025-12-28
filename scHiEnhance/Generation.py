import argparse
import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader, random_split
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
    def __init__(self, h5file, keys):
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
        return self.relu(self.fc2(x))


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
        tenkb = self.tenkb_enc(x[:, :self.tenkb_enc.fc1.in_features]).unsqueeze(1)
        ab = self.ab_enc(x[:, self.tenkb_enc.fc1.in_features:]).unsqueeze(1)
        h = torch.cat([tenkb, ab], dim=1).unsqueeze(3)
        h = self.combine(h).flatten(start_dim=1)
        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_var(h)) + 1e-8
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.tenkb_dec = TenKb_Decoder(tenkb_dim, z_dim, hidden_dim_1)
        self.ab_dec = AB_Decoder(ab_dim, z_dim, hidden_dim_2)

    def forward(self, z):
        return torch.cat(
            [self.tenkb_dec(z), self.ab_dec(z)], dim=1
        )


class VAE(nn.Module):
    def __init__(self, tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.encoder = Encoder(tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2)
        self.decoder = Decoder(tenkb_dim, ab_dim, z_dim, hidden_dim_1, hidden_dim_2)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generation from POE latent space"
    )

    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where patch-wise representations are stored"
    )
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save generated samples")

    parser.add_argument("--tenkb_dim", type=int, default=40512,
                        help="10kb feature dimension")
    parser.add_argument("--ab_dim", type=int, default=2376,
                        help="AB feature dimension")
    parser.add_argument("--z_dim", type=int, default=32,
                        help="Latent dimension")

    parser.add_argument("--hidden_dim_1", type=int, default=3000,
                        help="Hidden dim for 10kb branch")
    parser.add_argument("--hidden_dim_2", type=int, default=256,
                        help="Hidden dim for AB branch")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num_generate", type=int, default=10,
                        help="Number of generated samples per cell")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA if available")

    return parser.parse_args()



def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    h5 = h5py.File(args.dir + "/poe_impu_rx.hdf5", "r")
    keys = [k for k in h5.keys() if k != "sum"]

    dataset = DataSet(h5, keys)
    loader = DataLoaderX(
        dataset, batch_size=args.batch_size, num_workers=4
    )

    vae = torch.load(args.dir + '/cell_wise_model.pth', map_location=device)
    vae.eval()

    for n in range(args.num_generate):
        out_path = os.path.join(
            args.output_dir, f"generation_rx_{args.num_generate}_{n}.hdf5"
        )
        out_h5 = h5py.File(out_path, "a")

        for x, key in tqdm(loader):
            x = x.to(device)
            mu, sigma = vae.encoder(x)

            z = Normal(mu, sigma).sample()
            with torch.no_grad():
                rx = vae.decoder(z)

            rx = rx.cpu().numpy()
            z = z.cpu().numpy()

            for i, k in enumerate(key):
                grp = out_h5.create_group(f"{k}_{n}")
                grp.create_dataset("rx", data=rx[i], compression="gzip")
                grp.create_dataset("z", data=z[i], compression="gzip")

        out_h5.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)

