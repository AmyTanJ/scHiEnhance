import argparse
import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
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


def custom_sort(item):
    sp = item.split("/")
    a = int(sp[0])
    b = int(sp[1]) if len(sp) > 1 and sp[1].isdigit() else 1e9
    return a, b


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_conditions):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(200 * 200 + num_conditions, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, z_dim * 2),
        )

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

def reconstruct_chromosome(
    vae,
    rx_h5,
    input_h5,
    output_h5_path,
    key_list,
    patch_ids,
    rec_images,
    size_list,
    chosen_chr,
    z_dim,
    ab_dim,
    device,
):
    vae.eval()
    out_h5 = h5py.File(output_h5_path, "a")

    for key in tqdm(key_list):
        rx = rx_h5[key][:-ab_dim]
        rx = rx.reshape(-1, z_dim * 2)

        mu = torch.from_numpy(rx[:, :z_dim]).float().to(device)
        sigma = torch.from_numpy(rx[:, z_dim:]).float().to(device)
        z = Normal(mu, sigma).rsample()

        with torch.no_grad():
            rxrx = vae.decoder(z, patch_ids)

        rxrx = rxrx.cpu().numpy()

        width = int(eval(input_h5[key][str(chosen_chr)]["0"]["label"][:][2]))

        full = np.zeros((width, width), dtype=np.float32)
        count = np.zeros_like(full)

        idx_range = size_list[chosen_chr - 1]
        patches = rxrx[idx_range[0] : idx_range[1]]

        for p, patch in enumerate(patches):
            iw = int(rec_images[p].split("/")[-1])
            sub = patch.reshape(200, 200)
            for i in range(200):
                full[iw + i, iw + i : iw + i + 200] += sub[i]
                count[iw + i, iw + i : iw + i + 200] += 1

        full = np.divide(full, count, out=np.zeros_like(full), where=count > 0)

        csr = csr_matrix(full)
        grp = out_h5.create_group(key)
        grp.create_dataset(f"{chosen_chr}/data", data=csr.data, compression="gzip")
        grp.create_dataset(f"{chosen_chr}/indices", data=csr.indices, compression="gzip")
        grp.create_dataset(f"{chosen_chr}/indptr", data=csr.indptr, compression="gzip")
        grp.attrs["shape"] = full.shape

    out_h5.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch-wise reconstruction to full Hi-C matrix"
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where both high resolution intrachromosomal Hi-C interaction matrices are stored"
    )

    parser.add_argument("--num_conditions", type=int, default=1266,
                        help="Number of patches per chromosome")
    parser.add_argument("--z_dim", type=int, default=16,
                        help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=2000,
                        help="Hidden dimension")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA")

    return parser.parse_args()


def main(args):
    set_seed(42)

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    vae = torch.load(args.dir + '/finetuned_patch_wise_VAE.pth', map_location=device)

    rx_h5 = h5py.File(args.dir + '/poe_impu_rx.hdf5', "r")
    input_h5 = h5py.File(args.dir+'/h5_hic_dataset.hdf5', "r")

    ab = np.load(args.AB_dir + 'scAB_compartment.npy')
    ab_dim = ab.shape[1]

    example = list(input_h5.keys())[0]
    im_order = []
    input_h5[example].visititems(
        lambda n, o: im_order.append("/".join(n.split("/")[:2]))
        if len(n.split("/")) == 3 and n.split("/")[-1] == "img" else None
    )
    im_order = sorted(set(im_order), key=custom_sort)

    patch_ids = []
    for im in im_order:
        patch_ids.append(input_h5[example][im]["patch_id"][:])
    patch_ids = torch.from_numpy(
        np.asarray(patch_ids).reshape(args.num_conditions, args.num_conditions)
    ).float().to(device)

    size_list = [
        (0, 101), (101, 195), (195, 278), (278, 359), (359, 437),
        (437, 514), (514, 589), (589, 656), (656, 720), (720, 787),
        (787, 850), (850, 912), (912, 974), (974, 1038), (1038, 1091),
        (1091, 1141), (1141, 1189), (1189, 1235), (1235, 1266)
    ]

    keys = list(rx_h5.keys())

    for chr_id in range(1, 19):
        rec_images = [
            im for im in im_order if im.split("/")[1] == str(chr_id)
        ]
        out_path = os.path.join(
            args.dir, f"imputed_chr{chr_id}.hdf5"
        )
        reconstruct_chromosome(vae, rx_h5, input_h5, out_path, keys, patch_ids, rec_images, size_list, chr_id, args.z_dim, ab_dim, device)


if __name__ == "__main__":
    args = parse_args()
    main(args)
