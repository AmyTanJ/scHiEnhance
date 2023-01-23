import os
import numpy as np
from itertools import cycle, islice
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import h5py
from umap import UMAP
from tqdm import tqdm

def vis(args):
    """
    Clustering and calculating ARI score.
    """
    count = 0
    index = []
    H5File = h5py.File(args.dir + '/poe_z.hdf5', 'r+')
    for key in H5File.keys():
        index.append(H5File[key].name)
    for key in H5File.keys():
        if count == 0:
            H5File.create_dataset('sum', data = H5File[key][:], compression="gzip", maxshape=(None,))
            count+= 1
        else:
            H5File['sum'].resize((H5File['sum'].shape[0] + H5File[key][:].shape[0]), axis=0)
            H5File['sum'][-H5File[key][:].shape[0]:] = H5File[key][:]
    # UMAP
    umap = UMAP(n_components=2, args.n_neighbors, random_state=0).fit_transform((H5File['sum'][:]).reshape(args.cell_num,-1))
    type_path = args.cell_type
    cell_type = []
    cell_list = []
    count = 0
    for c in os.listdir(type_path):
        cell_type.append([c.split('.')[0], count])
        count+=1

    for t in index:
        cell_name = "_".join(t.split('_')[1:3])
        for i in os.listdir(type_path):
            file = open(os.path.join(type_path, i))
            for line in file.readlines():
                if line.split("\t")[0] == cell_name:
                    for ty in cell_type:
                        if ty[0] == i.split('.')[0]:
                            cell_list.append([cell_name, ty[1]])

    type2_list = []
    for i in range(len(cell_list)):
        type2_list.append(cell_list[i][1])
        
    # Calculate ARI score     
    pred = KMeans(n_clusters=len(np.unique(type2_list)), n_init = 200).fit((H5File['sum'][:]).reshape(args.cell_num,-1)).labels_
    H5File.close()
    ari2 = adjusted_rand_score(type2_list, pred)
    # legend color and order based on Dip-C paper
    color = np.array(list(islice(cycle(['darkgrey','blue','red','gold','salmon','blueviolet','steelblue',
                                        'orange','skyblue','limegreen','black','cyan','cornflowerblue','dodgerblue']), 100)))
    ctlist = ['Unknown', 'Interneuron', 'Adult Astrocyte', 'Oligodendrocyte Progenitor', 'Neonatal Astrocytee',
              'Medium Spiny Neuron','Hippocampal Granule Cell', 'Mature Oligodendrocyte', 'Cortical L2–5 Pyramidal_Cell',
             'Neonatal Neuron 1', 'Microglia Etc', 'Neonatal Neuron 2', 'Hippocampal Pyramidal Cell',
             'Cortical_L6_Pyramidal_Cell']
    type2_order = [13,5,10,2,9,8,11,12,0,3,4,6,7,1]
    type2_order = np.array(type2_order)
    
    # UMAP visualization
    plt.figure()
    for j,i in enumerate(type2_order):
        cell = (type2_list == i)
        plt.scatter(umap[cell, 0], umap[cell, 1],c=color[j], s=20, cmap='Spectral',label = ctlist[j],edgecolors='none', alpha=0.8)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [9,11,8,13,12,6,1,5,4,2,3,7,10,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],markerscale=1, prop={'size': 10}, bbox_to_anchor=(1,1), loc='upper left', fontsize=20)
    plt.title('UMAP projection of the scHi-C')
    plt.xlabel('ARI = {0}'.format(ari2))
    plt.savefig(args.dir + "/clustering.pdf", dpi=200, bbox_inches='tight')
    return plt.show()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True,
        help="root directory where cell-wise combined representations from POE framework are stored"
    )
    parser.add_argument(
        '--cell_type', type=str, required=True,
            help="file path containing cell type information"
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=25, help="k neighours of UMAP"
    )
    parser.add_argument(
        "--cell_num", type=int, default=1954, help="total cell number of scHi-C dataset"
    )
    args = parser.parse_args()

    vis(args)
