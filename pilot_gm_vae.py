"""
---------------------------------------------------------------------
-- Author: Mehdi Joodaki
---------------------------------------------------------------------

PILOT-GM-VAE

"""
from pilotpy.tools import *
from pilotpy.plot import *
import scanpy as sc
import time
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import random
import numpy as np
import torch
import time
from anndata import AnnData 
import os
import scipy.linalg as spl
import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange
import scipy.linalg
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.covariance import LedoitWolf
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from scipy.io import loadmat
from model.GMVAE import *
from model.GMVAE import GMVAE
import matplotlib.pyplot as plt
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from argparse import Namespace
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad

def train_gmvae(
    adata,
    dataset_name,
    pca_key='X_pca',
    labels_column=None,
    train_proportion=0.8,
    batch_size=32,
    batch_size_val=200,
    seed=1,
    epochs=50,
    learning_rate=1e-3,
    decay_epoch=-1,
    lr_decay=0.5,
    gaussian_size=64,
    num_classes=11,
    input_size=None,
    init_temp=1.0,
    decay_temp=1,
    hard_gumbel=0,
    min_temp=0.5,
    decay_temp_rate=0.013862944,
    w_gauss=1.0,
    w_categ=1.0,
    w_rec=2.0,
    rec_type="mse",
    cuda=0,
    gpuID=0,
    verbose=0,
    save_model=True,
    load_weights=False,
    apply_gmm=True,
):
    """
    Train or load a GMVAE model, save it to a folder, and optionally perform inference.
    
    Parameters:
        adata (AnnData): The input data object.
        dataset_name (str): Name of the dataset for saving the model.
        pca_key (str): Key in `adata.obsm` for the PCA-transformed data.
        labels_column (str or None): Column name in `adata.obs` for labels.
        train_proportion (float): Proportion of data for training.
        batch_size (int): Training batch size.
        batch_size_val (int): Validation and test batch size.
        seed (int): Random seed.
        epochs (int): Total number of epochs for training.
        learning_rate (float): Learning rate for training.
        decay_epoch (int): Reduces learning rate every decay_epoch epochs.
        lr_decay (float): Learning rate decay factor.
        gaussian_size (int): Size of the Gaussian latent space.
        num_classes (int): Number of classes.
        input_size (int or None): Input size (e.g., PCA dimension); inferred if None.
        init_temp (float): Initial temperature for Gumbel-Softmax.
        decay_temp (int): Flag to decay Gumbel temperature each epoch.
        hard_gumbel (int): Flag for using hard Gumbel-Softmax.
        min_temp (float): Minimum Gumbel temperature after annealing.
        decay_temp_rate (float): Rate of temperature decay.
        w_gauss (float): Weight for Gaussian loss.
        w_categ (float): Weight for categorical loss.
        w_rec (float): Weight for reconstruction loss.
        rec_type (str): Type of reconstruction loss ('bce' or 'mse').
        cuda (bool): Whether to use GPU.
        gpuID (int): ID of the GPU to use.
        verbose (int): Verbosity level.
        save_model (bool): Save the model after training.
        load_weights (bool): Load pre-trained weights if available.
        apply_gmm (bool): applying gmm on weights.
    
    Returns:
       None.

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
      torch.cuda.manual_seed(seed)

    data = adata.obsm[pca_key]
    input_size = data.shape[1] if input_size is None else input_size
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    
    # Extract labels if provided
    if labels_column and labels_column in adata.obs:
        labels = adata.obs[labels_column].astype('category').cat.codes
        labels_tensor = torch.tensor(labels.values, dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)  # Dataset with data and labels
    else:
        labels_tensor = None
        # Create a dataset with only data (no labels)
        dataset = TensorDataset(data_tensor)  # Only data

    
    # Extract labels if provided
    if labels_column and labels_column in adata.obs:
        labels = adata.obs[labels_column].astype('category').cat.codes
        labels_tensor = torch.tensor(labels.values, dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)
    else:
        labels_tensor = torch.zeros(len(data_tensor), dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)
    
   
    # Function to partition dataset into train, validation, and test
    def partition_dataset(dataset, train_proportion=train_proportion):
        n = len(dataset)
        train_num = int(n * train_proportion)
        indices = np.random.permutation(n)
        train_indices, val_indices = indices[:train_num], indices[train_num:]
        return train_indices, val_indices
    
    
    train_indices, val_indices = partition_dataset(dataset, train_proportion)
    test_indices = val_indices[:len(val_indices) // 2]
    val_indices = val_indices[len(val_indices) // 2:]
    
    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size_val, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size_val, sampler=SubsetRandomSampler(test_indices))
    whole_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Prepare arguments for GMVAE
    args = Namespace(
        dataset=dataset_name,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        batch_size_val=batch_size_val,
        learning_rate=learning_rate,
        decay_epoch=decay_epoch,
        lr_decay=lr_decay,
        gaussian_size=gaussian_size,
        num_classes=num_classes,
        input_size=input_size,
        init_temp=init_temp,
        decay_temp=decay_temp,
        hard_gumbel=hard_gumbel,
        min_temp=min_temp,
        decay_temp_rate=decay_temp_rate,
        w_gauss=w_gauss,
        w_categ=w_categ,
        w_rec=w_rec,
        rec_type=rec_type,
        cuda=cuda,
        labels_column=labels_column,
        verbose=verbose,
    )
   
    # Initialize GMVAE
    gmvae = GMVAE(args)
    
    # Model directory and path
    model_dir = f"./trained_models/{dataset_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gmvae_weights.pth")
    
    # Load pre-trained weights or train the model
    if load_weights:
        gmvae.network.load_state_dict(torch.load(model_path))
        gmvae.network.eval()  # Set the model to evaluation mode
        print(f"Loaded pre-trained weights from {model_path}.")
    else:
        # Train the model
        print("Training the GMVAE model...")
        gmvae.train(train_loader, val_loader)
        if save_model:
            torch.save(gmvae.network.state_dict(), model_path)
            print(f"Saved trained model weights to {model_path}.")
   
    print("Performing inference...")
    z_latent, x_recon, cluster_probs, clusters = gmvae.infer(whole_loader)
    if apply_gmm:
        gmm_clusters = GaussianMixture(n_components=num_classes)
        cluster_assignments = gmm_clusters.fit_predict(cluster_probs)
        adata.obs['component_assignment'] = cluster_assignments+1
        adata.obs['component_assignment'] = adata.obs['component_assignment'].astype(str)
    adata.obsm['z_laten'] = z_latent
    adata.obsm['weights'] = cluster_probs
    adata.obsm['x_prim']=x_recon
    adata.obs['cluster_assignments_by_model_before_gmm']=cluster_assignments
    print("Done!")
   






def plot_umap_and_stacked_bar(
    adata, 
    cell_type_col, 
    component_col='component_assignment', 
    palette_name=None, 
    title="Cell Type vs Component Assignment", 
    xlabel="Component Assignment", 
    ylabel="Proportion", 
    save_path='figures/Bar_plot.png', 
    umap_save_name="_components_vs_cell_types.png", 
    umap_size=5, 
    umap_legend_fontsize=10, 
    umap_ncols=2, 
    umap_wspace=0.55, 
    bar_figsize=(14, 8), 
    umap_figsize=(10, 10), 
    axes_titlesize=18, 
    axes_labelsize=16, 
    legend_fontsize=14, 
    legend_bbox_to_anchor=(1.05, 1), 
    legend_loc='upper left', 
    legend_borderaxespad=0.0
):
    """
    Plots a UMAP and a normalized stacked bar chart of component assignments vs. cell types.

    Parameters:
        adata: Input AnnData object.
        cell_type_col (str): Name of the column containing cell type annotations.
        component_col (str): Name of the column containing component assignments.
        palette_name (str, optional): Name of the palette or a custom palette list to use for coloring.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        save_path (str, optional): File path to save the bar plot (PDF or PNG). If None, the plot is not saved.
        umap_save_name (str, optional): File name for saving the UMAP plot.
        umap_size (int, optional): Point size for the UMAP plot.
        umap_legend_fontsize (int, optional): Font size for the UMAP legend.
        umap_ncols (int, optional): Number of columns for the UMAP plots.
        umap_wspace (float, optional): Width space between UMAP plots.
        bar_figsize (tuple, optional): Figure size for the bar plot.
        umap_figsize (tuple, optional): Figure size for the UMAP plot.
        axes_titlesize (int, optional): Font size for axes titles.
        axes_labelsize (int, optional): Font size for axes labels.
        legend_fontsize (int, optional): Font size for the legend.
        legend_bbox_to_anchor (tuple, optional): Anchor position for the legend.
        legend_loc (str, optional): Location of the legend.
        legend_borderaxespad (float, optional): Padding between legend and axes border.

    Returns:
        None
    """
    # Ensure component_col is treated as a string
    #adata.obs[component_col] = adata.obs[component_col].astype(str)

    # Set default palette if none is provided
    adata.obs['component_assignment'] = adata.obs['component_assignment'].astype(str)
    if palette_name is None:
        palette = sns.color_palette("Set1", 9) + sns.color_palette("Set2", 8) + sns.color_palette("Set3", 12) + sns.color_palette("Dark2", 4)
        palette = palette[:33]  # Limit the palette to exactly 33 colors
    else:
        palette = palette_name

    # Plot UMAP
    sc.pl.umap(
        adata,
        color=[component_col, cell_type_col],
        save=umap_save_name,  
        ncols=umap_ncols, 
        wspace=umap_wspace, 
        legend_fontsize=umap_legend_fontsize, 
        size=umap_size, 
        palette=palette
    )

    # Set font properties globally
    plt.rc('axes', titlesize=axes_titlesize)  # Set font size for axes titles
    plt.rc('axes', labelsize=axes_labelsize)  # Set font size for axes labels
    plt.rc('legend', fontsize=legend_fontsize)  # Set font size for legend

    # Prepare data for the stacked bar plot
    df = pd.DataFrame({
        'component_assignment': adata.obs[component_col],
        'Cell Type': adata.obs[cell_type_col]
    })

    # Create a cross-tab to get counts (no normalization)
    cross_tab = pd.crosstab(df[component_col], df['Cell Type'])

    # Normalize each row to ensure bars have the same total length
    cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0)

    # Plot as a stacked bar plot using the custom palette (with normalization)
    plt.figure(figsize=bar_figsize)  # Use dynamic figure size for bar plot
    ax = cross_tab_normalized.plot(kind='bar', stacked=True, color=palette[:33], figsize=bar_figsize)

    # Add labels and titles
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Move legend to the right side of the plot
    plt.legend(
        title='Cell Type', 
        bbox_to_anchor=legend_bbox_to_anchor, 
        loc=legend_loc, 
        borderaxespad=legend_borderaxespad
    )

    # Save the plot as a file if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()


def compute_distance(k, l, m_s, m_t, C_s, C_t, covariance_type, log=False,epsilon = 1e-3):
    
    """
    Computes the Bures-Wasserstein distance between components k and l of GMMs.

    Args:
    - k (int): 
        Index of the component in GMM s for which the distance is computed.

    - l (int): 
        Index of the component in GMM t for which the distance is computed.

    - m_s (numpy.ndarray): 
        Mean vectors of GMM s with shape [num_components_s, dim], where each row corresponds to the mean vector of a GMM component.

    - m_t (numpy.ndarray): 
        Mean vectors of GMM t with shape [num_components_t, dim], where each row corresponds to the mean vector of a GMM component.

    - C_s (numpy.ndarray): 
        Covariance matrices of GMM s. Shape is [num_components_s, dim] for diag covariance or 
        [num_components_s, dim, dim] for full covariance.

    - C_t (numpy.ndarray): 
        Covariance matrices of GMM t. Shape is [num_components_t, dim] for diag covariance or 
        [num_components_t, dim, dim] for full covariance.

    - covariance_type (str): 
        Specifies the type of covariance matrix. Accepted values are:
        - 'diag': Indicates diagonal covariance matrices.
        - 'full': Indicates full covariance matrices.

    - log (bool, optional): 
        If True, enables logging of intermediate steps in the Bures-Wasserstein computation. Default is False.

    - epsilon (float, optional): 
        A small positive value added to the diagonal elements of covariance matrices to ensure numerical stability. 
        Default is 1e-3.

    Returns:
    - float: 
        The Bures-Wasserstein distance between component k of GMM s and component l of GMM t.

    Raises:
    - ValueError: 
        If covariance_type is not 'diag' or 'full'.
    
    - RuntimeError: 
        If the computed Bures-Wasserstein distance returns NaN values after multiple attempts.

    Notes:
    - If the covariance matrices are diagonal, they are converted into full diagonal matrices.
    - If numerical instability is detected (NaN values in the computed distance), epsilon is reduced iteratively to stabilize computations.
    """
    
    ms_k = m_s[k, :]  # Mean vector of component k in GMM s
    mt_l = m_t[l, :]  # Mean vector of component l in GMM t

    if covariance_type == 'diag': 
        
        Cs_k = np.diag(C_s[k, :][k])   # Convert diagonal elements into a diagonal covariance matrix
        Ct_l = np.diag(C_t[l, :][l])  # Convert diagonal elements into a diagonal covariance matrix
         
    else:
        # Full covariance case
        Cs_k = C_s[k, :, :]  # Covariance matrix of component k in GMM s
        Ct_l = C_t[l, :, :]  # Covariance matrix of component l in GMM t
        Cs_k += epsilon * np.eye(Cs_k.shape[0])
        Ct_l += epsilon * np.eye(Ct_l.shape[0])

    attempt=0
    max_attempts=10
    bures_wasserstein = ot.gaussian.bures_wasserstein_distance(ms_k, mt_l, Cs_k, Ct_l, log=log)
    while attempt < max_attempts:
        if np.isnan(bures_wasserstein):
            epsilon *= 0.5
            Cs_k += epsilon * np.eye(Cs_k.shape[0])
            Ct_l += epsilon * np.eye(Ct_l.shape[0])
            bures_wasserstein = ot.gaussian.bures_wasserstein_distance(ms_k, mt_l, Cs_k, Ct_l, log=log)
            attempt=attempt+1
        else:
            break
      
    
    return bures_wasserstein



def compute_emd(i, j, samples_id, EMD, adata, compute_distance, log, covariance_type, wass_dis,epsilon = 1e-3):
    
    """
    Computes the Earth Mover's Distance (EMD) between two GMM representations in single-cell data.

    Args:
    - i (int): 
        Index of the first sample in the `samples_id` list.

    - j (int): 
        Index of the second sample in the `samples_id` list.

    - samples_id (list): 
        List of sample identifiers.

    - EMD (numpy.ndarray): 
        A symmetric matrix storing precomputed EMD values.

    - adata (AnnData): 
        AnnData object containing `GMVAE_Representation` in `.uns`, where each sample has 
        means, covariances, and weights of the Gaussian Mixture Model (GMM).

    - compute_distance (function): 
        A function that computes pairwise distances between GMM components using a specified metric.

    - log (bool): 
        If True, enables logging of intermediate steps in distance computation.

    - covariance_type (str): 
        Specifies the type of covariance matrix. Accepted values are:
        - 'diag': Indicates diagonal covariance matrices.
        - 'full': Indicates full covariance matrices.
    - wass_dis (float): 
        Unused parameter (potentially for storing Wasserstein distance results).

    - epsilon (float, optional): 
        A small positive value added to the diagonal elements of covariance matrices to ensure numerical stability. 
        Default is 1e-3.

    Returns:
    - tuple or None: 
        If the computation is performed, returns a tuple (i, j, w_d), where:
        - `i` (int): Index of the first sample.
        - `j` (int): Index of the second sample.
        - `w_d` (float): Computed Earth Mover’s Distance (EMD) between the two samples.
        If `s == t` (same sample) or EMD is already computed, returns `None`.

    Notes:
    - If `s == t`, the function skips computation to avoid redundant calculations.
    - If EMD[i, j] is already computed, the function returns `None`.
    - Uses `Parallel(n_jobs=-1)` to compute pairwise component distances in parallel.
    - The resulting EMD is stored **symmetrically** in the matrix `EMD[i, j]` and `EMD[j, i]`.
    """

    
    s = samples_id[i]
    t = samples_id[j]
    if s == t:
        return None  # No need to compute for the same sample pair
    
    if EMD[i, j] != 0:
        return None  # Already computed this pair
    
    gmm_repr_s = adata.uns['GMVAE_Representation'][s]
    m_s = np.array(gmm_repr_s['means'])
    C_s = np.array(gmm_repr_s['covariances'])

    gmm_repr_t = adata.uns['GMVAE_Representation'][t]
    m_t = np.array(gmm_repr_t['means'])
    C_t = np.array(gmm_repr_t['covariances'])

    num_components_s = m_s.shape[0]
    num_components_t = m_t.shape[0]

    # Normalize weights
    weights1 = np.array(gmm_repr_s['weights']) / np.sum(gmm_repr_s['weights'])
    weights2 = np.array(gmm_repr_t['weights']) / np.sum(gmm_repr_t['weights'])

    # Compute distances in parallel for components
    distances_flat = Parallel(n_jobs=-1)(
        delayed(compute_distance)(k, l, m_s, m_t, C_s, C_t, covariance_type, log=log,epsilon =epsilon) 
        for k in range(num_components_s) for l in range(num_components_t)
    )
    distances = np.array(distances_flat).reshape(num_components_s, num_components_t)

    # Compute the EMD
    w_d = ot.emd2(weights1, weights2, distances, numThreads=50)

    # Store results symmetrically in the matrix
    EMD[i, j] = w_d
    EMD[j, i] = w_d

    return (i, j, w_d)  # Return the result to update the matrix later



def gmmvae_wasserstein_distance(adata,emb_matrix='X_PCA',
clusters_col='component_assignment',sample_col='sampleID',status='status',
                              metric='cosine',
                               regulizer=0.2,normalization=True,
                               regularized='unreg',reg=0.1,
                               res=0.01,steper=0.01,data_type='scRNA',
                                return_sil_ari=False,num_components=4,random_state=2,covariance_type='full',wass_dis=True,epsilon = 1e-4,log=False,apply_gmm=False):
    
    
    """
    Computes the GMMVAE-based Wasserstein distance between samples using the Earth Mover's Distance (EMD)
    and Gaussian Mixture Variational Autoencoder (GMVAE) representations.

    Args:
    - adata (AnnData): 
        An AnnData object containing scRNA-seq or other omics data.

    - emb_matrix (str, optional): 
        The key in `adata.obsm` representing the embedding matrix. Default is 'X_PCA'.

    - clusters_col (str, optional): 
        The column name in `adata.obs` that contains cluster assignments. Default is 'component_assignment'.

    - sample_col (str, optional): 
        The column name in `adata.obs` that contains sample IDs. Default is 'sampleID'.

    - status (str, optional): 
        The column name in `adata.obs` that represents metadata status (e.g., disease vs. control). Default is 'status'.

    - metric (str, optional): 
        The distance metric to be used in clustering. Default is 'cosine'.

    - regulizer (float, optional): 
        Regularization strength for GMVAE. Default is 0.2.

    - normalization (bool, optional): 
        Whether to normalize data before clustering. Default is True.

    - regularized (str, optional): 
        Type of regularization ('unreg' or other). Default is 'unreg'.

    - reg (float, optional): 
        Additional regularization parameter. Default is 0.1.

    - res (float, optional): 
        Resolution parameter for clustering. Default is 0.01.

    - steper (float, optional): 
        Step size for clustering. Default is 0.01.

    - data_type (str, optional): 
        Type of data being analyzed ('scRNA' or other). Default is 'scRNA'.

    - return_sil_ari (bool, optional): 
        Whether to compute silhouette and ARI scores for clustering. Default is False.

    - num_components (int, optional): 
        Number of Gaussian components in GMVAE. Default is 4.

    - random_state (int, optional): 
        Random seed for reproducibility. Default is 2.

    - covariance_type (str, optional): 
        Type of covariance used in GMM ('full' or 'diag'). Default is 'full'.

    - wass_dis (bool, optional): 
        Whether to compute Wasserstein distance. Default is True.

    - epsilon (float, optional): 
        Small numerical stability parameter. Default is 1e-4.

    - log (bool, optional): 
        If True, enables logging for distance computation. Default is False.

    - apply_gmm (bool, optional): 
        Whether to apply GMM for clustering. Default is False.

    Returns:
    - None: 
        Stores computed **Wasserstein distances (EMD)** and **clustering results** in `adata.uns`.

    Notes:
    - Extracts PCA or raw data based on `data_type` and prepares it for GMVAE clustering.
    - Computes Gaussian Mixture Variational Autoencoder (GMVAE) representations and stores them in `adata.uns`.
    - Uses **Earth Mover’s Distance (EMD)** to measure distances between GMM components across samples.
    - Stores **EMD matrix and clustering results** in `adata.uns`.
    """



    if data_type=='scRNA':
        data,annot=extract_data_anno_scRNA_from_h5ad(adata,emb_matrix=emb_matrix,
        clusters_col=clusters_col,sample_col=sample_col,status=status)
        pca_results_df = pd.DataFrame(adata.obsm[emb_matrix]).reset_index(drop=True)
    else:
        data,annot=extract_data_anno_pathomics_from_h5ad(adata,var_names=list(adata.var_names),clusters_col=clusters_col,sample_col=sample_col,status=status)
        pca_results_df = pd.DataFrame(adata.X).reset_index(drop=True)
       
    
    adata.uns['annot']=annot
    sample_ids = adata.obs[sample_col].reset_index(drop=True)
    cell_subtypes = adata.obs[clusters_col].reset_index(drop=True)
    status = adata.obs[status].reset_index(drop=True)
    
        # Concatenate the PCA results with 'sampleID', 'cell_subtype', and 'status'
    combined_pca_df = pd.concat([pca_results_df, sample_ids, cell_subtypes, status], axis=1)
  
    #combined_pca_df = combined_pca_df.rename(columns={clusters_col: 'cell_types',Status:'status'  }) 
                                            
    current_columns = combined_pca_df.columns

    # Create a mapping for the last three columns
    rename_dict = {current_columns[-3]: 'sampleID', 
                   current_columns[-2]: 'cell_type', 
                   current_columns[-1]: 'status'}
    
    # Rename the columns using the dictionary
    combined_pca_df.rename(columns=rename_dict, inplace=True)                                                                                            
        #combined_pca_df.columns[-3:]=['cell_type','sampleID','status']
    adata.uns['Datafame_for_use'] = combined_pca_df
    if wass_dis:
        gaussian_mixture_vae_representation(adata,num_components=num_components,sample_col=sample_col,covariance_type=covariance_type,random_state=random_state,apply_gmm=apply_gmm)
        samples_id = list(adata.uns['GMVAE_Representation'].keys())
        n_samples = len(samples_id)
        EMD = np.zeros((n_samples, n_samples))
        
           # Extract means and covariances
        samples_id = list(adata.uns['GMVAE_Representation'].keys())
        n_samples = len(samples_id)
        EMD = np.zeros((n_samples, n_samples))
    #########################

    start_time = time.time()
    if wass_dis:
        
        start_time = time.time()
        # Parallelize the outer loops
        results = Parallel(n_jobs=-1)(
            delayed(compute_emd)(i, j, samples_id, EMD, adata, compute_distance, log, covariance_type,wass_dis,epsilon=epsilon)
            for i in range(n_samples) for j in range(i + 1, n_samples)  # Only compute for j > i to avoid duplicates
        )

        # Update the EMD matrix with the computed distances
        for rest in results:
            if rest is not None:
                i, j, w_d = rest
                EMD[i, j] = w_d
                EMD[j, i] = w_d
   
    
        emd_df = pd.DataFrame.from_dict(EMD).T
        emd_df.columns=samples_id 
        emd_df['sampleID']=samples_id 
        emd_df=emd_df.set_index('sampleID')
        adata.uns['EMD_df']=emd_df
        adata.uns['EMD'] =EMD


    else:
        EMD=adata.uns['EMD']
        
   
    #Computing clusters and then ARI
    if return_sil_ari:
        predicted_labels, ARI, real_labels = Clustering(EMD, annot,metric=metric,res=res,steper=steper)
        adata.uns['real_labels'] =real_labels
        #Computing Sil
        Silhouette = Sil_computing(EMD, real_labels,metric=metric)
        adata.uns['Sil']=Silhouette
        adata.uns['ARI']=ARI
    else:
        adata.uns['real_labels']=return_real_labels(annot)
        
    elapsed_time = time.time() - start_time
    annot= adata.uns['annot']
    annot[annot.columns[0]]=list(adata.obs[clusters_col])
    proportions = Cluster_Representations(annot)
    adata.uns['proportions'] = proportions
    #print(f"Time taken for the ot part: {elapsed_time:.2f} seconds")

      

def gaussian_mixture_vae_representation(adata, num_components=5,patience=10,sample_col='sampleID',
                                        covariance_type='full',
                                       apply_gmm=True,random_state=0):
    """
    Computes the Gaussian Mixture Variational Autoencoder (GMVAE) representation for each sample in an AnnData object.

    Args:
    - adata (AnnData): 
        An AnnData object containing transcriptomic data and precomputed PCA embeddings.

    - num_components (int, optional): 
        Number of Gaussian components in the mixture model. Default is 5.

    - patience (int, optional): 
        Number of epochs for early stopping if no improvement. Default is 10.

    - sample_col (str, optional): 
        The column name in `adata.obs` that contains sample IDs. Default is 'sampleID'.

    - covariance_type (str, optional): 
        Specifies the type of covariance matrix. Accepted values:
        - 'diag': Diagonal covariance matrices.
        - 'full': Full covariance matrices.
        Default is 'full'.
    - apply_gmm (bool, optional): 
        Whether to apply Gaussian Mixture Model (GMM) for clustering. Default is False.

    - random_state (int, optional): 
        Random seed for reproducibility. Default is 0.

    Returns:
    - AnnData: 
        The modified `adata` object with updated fields:
        - `adata.obs['component_assignment']`: Assigned components for each cell.
        - `adata.uns['proportions']`: Cluster proportions per sample.
        - `adata.uns['GMVAE_Representation']`: Dictionary containing means, covariances, and weights of the GMM components.

    Notes:
    - The function extracts **PCA-reduced features** and groups cells into **Gaussian mixture components**.
    - If `apply_gmm=True`, a **GMM model** is fit on the learned latent space.
    - Computes and stores:
      - **Means** and **covariances** for each Gaussian component.
      - **Mixing weights** representing cluster proportions.
    - The `GMVAE_Representation` dictionary stores the **GMM parameters** per sample.
    """

    df = adata.uns['Datafame_for_use']
    df = df[df['cell_type'] != 'Unknown']
    df = df.drop(['status'], axis=1)
    df = df.drop(['cell_type'], axis=1)
    pca_data = df.drop(['sampleID'], axis=1).values  # Assume PCA features are here
    data_tensor = torch.tensor(pca_data, dtype=torch.float32)  # Convert to tensor

    input_dim = pca_data.shape[1]
    sources = df['sampleID'].unique()

   
   
    #z_latent, reconstructed_data, weights, cluster_assignments = gmvae.infer(test_loader)

   # adata.obsm['z_laten'] = z_latent
    weights=adata.obsm['weights']
    #adata.obsm['x_prim']=reconstructed_data
    #adata.obs['cluster_assignments_by_model_before_gmm']=cluster_assignments
    adata.obs['component_assignment'] = np.nan
    if apply_gmm:
        gmm_clusters = GaussianMixture(n_components=num_components, covariance_type='full', random_state=random_state)
        cluster_assignments = gmm_clusters.fit_predict(weights)
    # Calculate means and covariances for each sample
    params = {}
    for source in sources:
        data = df[df['sampleID'] == source]
        data_values = data.drop(['sampleID'], axis=1).values
        data_tensor_source = torch.tensor(data_values, dtype=torch.float32)

        # Get indices as a list for proper indexing
        source_indices = data.index.tolist()

        if source_indices:

            component_assignments = cluster_assignments[source_indices]
            adata_obs_indices = adata.obs.index[adata.obs[sample_col] == source].tolist()
            adata.obs.loc[adata_obs_indices, 'component_assignment'] = component_assignments
            # Initialize arrays to store means and covariances for each component
            means = np.zeros((num_components, input_dim))
            covariances = np.zeros((num_components, input_dim, input_dim))

            # Calculate means and covariances for each component
            for k in range(num_components):
                # Get the indices of the cells assigned to component k
                indices = component_assignments == k
                assigned_data = data_tensor_source[indices]

                if assigned_data.size(0) > 0:  # Ensure there are assigned points
                    means[k] = assigned_data.mean(dim=0).detach().numpy()
                    if covariance_type=='full':
                        covariances[k] = np.cov(assigned_data.numpy(), rowvar=False)
                    else:

                        covariances[k] = np.var(assigned_data.numpy(), axis=0)
                        covariances[k] = np.diag(covariances[k])

            # Mixing weights (proportions) for this sample
            weights_sample = weights[source_indices].mean(axis=0)

           # weights_sample = weights[source_indices].mean(dim=0).detach().numpy()

            params[source] = {
                'means': means,
                'covariances': covariances,
                'weights': weights_sample,
                'proportion': len(data_values) / len(df)
            }

    annot= adata.uns['annot']
    annot[annot.columns[0]]=list(adata.obs['component_assignment'])
    proportions = Cluster_Representations(annot)
    adata.uns['proportions'] = proportions
    adata.uns['GMVAE_Representation'] = params

    return adata


def filter_cells_by_sample_and_cell_type(adata, sample_column, cell_type_column):
    """
    Randomly samples cells from each sample-cell type combination in the AnnData object
    based on specific rules.

    Rules:
        - If a sample-cell type combination has < 100 cells, retain all cells.
        - If a sample-cell type combination has between 100 and 1000 cells, sample 10%.
        - If a sample-cell type combination has > 1000 cells, sample 5%.

    Parameters:
        adata (AnnData): Input AnnData object.
        sample_column (str): Column name in `adata.obs` that identifies samples.
        cell_type_column (str): Column name in `adata.obs` that identifies cell types.

    Returns:
        AnnData: A filtered AnnData object containing the sampled cells.
    """
    import numpy as np

    # Ensure reproducibility
    np.random.seed(42)

    # Initialize an empty list to collect cell indices
    selected_indices = []

    # Group by sample and cell type
    grouped = adata.obs.groupby([sample_column, cell_type_column])
    for (sample, cell_type), indices in grouped.indices.items():
        n_cells = len(indices)

        # Apply sampling rules
        if n_cells < 100:
            sampled_indices = indices  # Retain all cells
        elif 100 <= n_cells <= 1000:
            n_sample = max(1, int(n_cells * 0.1))  # Sample 10%
            sampled_indices = np.random.choice(indices, n_sample, replace=False)
        else:  # n_cells > 1000
            n_sample = max(1, int(n_cells * 0.05))  # Sample 5%
            sampled_indices = np.random.choice(indices, n_sample, replace=False)

        selected_indices.extend(sampled_indices)

    # Filter the AnnData object
    filtered_adata = adata[selected_indices].copy()

    # Display the number of cells in the filtered dataset
    print(f"Original number of cells: {adata.shape[0]}")
    print(f"Filtered number of cells: {filtered_adata.shape[0]}")

    return filtered_adata
  
    
def genes_selection_heatmap(
        adata:  ad.AnnData,
        cell_type: str,
        filter_table_feature: str = 'R-squared',
        filter_table_feature_pval: str = 'adjusted P-value',
        table_filter_thr: float = 0.05,
        table_filter_pval_thr: float = 0.05,
        cluster_method: str = 'complete',
        cluster_metric: str = 'correlation',
        scaler_value: float = 0.4,
        cmap_color: str = 'RdBu_r',
        figsize: tuple = (7, 9),
        fontsize: int = 14,
        path_to_results: str = 'Results_PILOT/',
        df=None,
     col_cluster=False,row_cluster=True,yticklabels=True, xticklabels=False, z_score=0,dendrogram_ratio=(0.2, 0),cbar_pos=(-0.08, 0.8, .03, .1),name_col_fil='gene',convert_names=False,save_path_heatmap: str = 'Results_PILOT/Markers/heatmap.pdf'
    ):
    """
    Generates a heatmap for selected genes based on hierarchical clustering of their expression curves over pseudotime.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object containing single-cell or similar data.

    cell_type : str
        The cell type for which the gene selection and clustering will be performed.

    filter_table_feature : str, optional
        Feature used to filter genes from the table. Default is 'R-squared'.

    filter_table_feature_pval : str, optional
        Feature representing the p-value used to filter genes. Default is 'adjusted P-value'.

    table_filter_thr : float, optional
        Threshold for filtering genes based on the selected feature. Default is 0.05.

    table_filter_pval_thr : float, optional
        Threshold for filtering genes based on the p-value feature. Default is 0.05.

    cluster_method : str, optional
        Method used for hierarchical clustering. Examples include 'single', 'complete', and 'average'. Default is 'complete'.

    cluster_metric : str, optional
        Distance metric used for clustering. Examples include 'correlation', 'euclidean', and 'cosine'. Default is 'correlation'.

    scaler_value : float, optional
        Scaling factor used during clustering. Default is 0.4.

    cmap_color : str, optional
        Colormap used for the heatmap. Default is 'RdBu_r'.

    figsize : tuple, optional
        Size of the figure for the heatmap. Default is (7, 9).

    fontsize : int, optional
        Font size for heatmap labels. Default is 14.

    path_to_results : str, optional
        Directory path where results, such as activity curves, will be saved. Default is 'Results_PILOT/'.

    df : pandas.DataFrame, optional
        Optional DataFrame containing additional gene information. Default is None.

    col_cluster : bool, optional
        Whether to cluster columns in the heatmap. Default is False.

    row_cluster : bool, optional
        Whether to cluster rows in the heatmap. Default is True.

    yticklabels : bool, optional
        Whether to display y-axis (row) labels in the heatmap. Default is True.

    xticklabels : bool, optional
        Whether to display x-axis (column) labels in the heatmap. Default is False.

    z_score : int, optional
        If 0 or 1, standardize genes or samples, respectively, in the heatmap. Default is 0.

    dendrogram_ratio : tuple, optional
        Ratio for the size of row and column dendrograms. Default is (0.2, 0).

    cbar_pos : tuple, optional
        Position of the color bar in the heatmap. Default is (-0.08, 0.8, .03, .1).

    name_col_fil : str, optional
        Name of the column in `df` used for filtering genes. Default is 'gene'.

    convert_names : bool, optional
        Whether to convert gene names during processing. Default is False.

    save_path_heatmap : str, optional
        File path to save the generated heatmap. Default is 'Results_PILOT/Markers/heatmap.pdf'.

    Returns
    -------
    None
        Results are stored in the `adata.uns` dictionary under the key `'gene_selection_heatmap'` with the following sub-keys:
        - `'cell_type'`: The cell type used for analysis.
        - `'curves'`: Filtered gene curves.
        - `'noised_curves'`: Noised gene curves after filtering.
        - `'pseudotime_sample_names'`: Pseudotime-sorted sample names.
        - `'curves_activities'`: Activity profiles of the filtered and clustered gene curves.

    Notes
    -----
    - The function filters genes based on specific thresholds for a given feature and p-value, performs hierarchical clustering, and plots a heatmap of clustered genes.
    - Intermediate results, such as gene curves and activity profiles, are saved in the specified results directory and `adata.uns`.

    """

    


    print("Filter genes ...")
    curves, noised_curves, pseudotime_sample_names = get_noised_curves(adata, cell_type,
                                                                       filter_table_feature,
                                                                       filter_table_feature_pval,
                                                                       table_filter_thr,
                                                                       table_filter_pval_thr,
                                                                       path_to_results,df=df,name_col_fil=name_col_fil)

    print("Cluster genes using hierarchical clustering... ")
    genes_clusters = cluster_genes_curves(noised_curves,
                                          cluster_method,
                                          cluster_metric,
                                          scaler_value)

    print("Compute curves activities... ")
    print("Save curves activities... ")
    curves_activities = compute_curves_activities(noised_curves, genes_clusters,
                              pseudotime_sample_names,
                              cell_type, path_to_results)

    print("Plot the heatmap of genes clustered... ")
    plot_heatmap_curves(noised_curves, genes_clusters,
                        cluster_method, cluster_metric,
                        figsize, fontsize,save_path=save_path_heatmap,df=df, col_cluster=col_cluster,row_cluster=row_cluster,yticklabels=yticklabels,  xticklabels=xticklabels,  z_score=z_score, cmap_color=cmap_color,dendrogram_ratio=dendrogram_ratio,cbar_pos=cbar_pos,convert_names=convert_names)
    
    adata.uns['gene_selection_heatmap'] = {'cell_type': cell_type,
                                           'curves': curves,
                                           'noised_curves': noised_curves,
                                           'pseudotime_sample_names': pseudotime_sample_names,
                                           'curves_activities': curves_activities}
def plot_heatmap_curves(curves: pd.DataFrame = None,
                        genes_clusters: pd.DataFrame = None,
                        cluster_method: str = 'complete',
                        cluster_metric: str = 'correlation',
                        figsize: tuple = (7, 9),
                        fontsize: int = 14,
                        save_path: str = 'Results_PILOT/Markers/heatmap.pdf',df=None, col_cluster=False, row_cluster=False,  yticklabels=True,  xticklabels=False,  z_score=0, cmap_color='RdBu_r',dendrogram_ratio=(0.2, 0),cbar_pos=(-0.08, 0.8, .03, .1),convert_names=False):
    
    """
    Plots a heatmap of clustered gene expression curves and saves it as a PDF file.

    Parameters
    ----------
    curves : pd.DataFrame
        DataFrame containing gene expression curves. Each row corresponds to a gene, and columns correspond to samples or pseudotime points.

    genes_clusters : pd.DataFrame
        DataFrame containing cluster information for each gene. Must include a column `'Gene ID'` for matching with `curves` 
        and a column `'cluster'` for cluster assignments.

    cluster_method : str, optional
        Method used for hierarchical clustering. Options include 'single', 'complete', 'average', etc. Default is 'complete'.

    cluster_metric : str, optional
        Metric used for computing distances in clustering. Options include 'correlation', 'euclidean', etc. Default is 'correlation'.

    cmap_color : str, optional
        Colormap for the heatmap. Default is 'RdBu_r'.

    figsize : tuple, optional
        Size of the figure (width, height). Default is (7, 9).

    fontsize : int, optional
        Font size for annotations and labels. Default is 14.

    save_path : str, optional
        Path to save the heatmap as a PDF. Default is 'Results_PILOT/Markers/heatmap.pdf'.

    df : pandas.DataFrame, optional
        DataFrame containing additional gene metadata, including a mapping from Ensembl IDs to gene names. Default is None.

    col_cluster : bool, optional
        Whether to cluster columns in the heatmap. Default is False.

    row_cluster : bool, optional
        Whether to cluster rows in the heatmap. Default is False.

    yticklabels : bool, optional
        Whether to display y-axis labels (gene names). Default is True.

    xticklabels : bool, optional
        Whether to display x-axis labels (sample or pseudotime names). Default is False.

    z_score : int, optional
        If 0 or 1, standardizes rows or columns, respectively, in the heatmap. Default is 0.

    dendrogram_ratio : tuple, optional
        Proportion of the figure used for row and column dendrograms. Default is (0.2, 0).

    cbar_pos : tuple, optional
        Position of the color bar in the heatmap. Default is (-0.08, 0.8, 0.03, 0.1).

    convert_names : bool, optional
        Whether to convert Ensembl IDs to gene names using the `df` DataFrame. Default is False.

    Returns
    -------
    None
        The function saves the heatmap as a PDF file at the specified `save_path`.

    Notes
    -----
    - The heatmap rows are optionally colored based on gene clusters provided in `genes_clusters`.
    - The function supports hierarchical clustering of rows and/or columns using specified methods and metrics.
    - If `convert_names` is True, Ensembl IDs in `curves` and `genes_clusters` will be replaced with gene names using the mapping provided in `df`.

    """
    
    if convert_names:
        gene_dict = dict(zip(df['Ens_ID'], df['gene']))
        curves.index = curves.index.map(gene_dict)
        genes_clusters['Gene ID'] = genes_clusters['Gene ID'].map(gene_dict)
    
    # Define row colors based on gene clusters
    
    my_palette = dict(zip(genes_clusters['cluster'].unique(),sns.color_palette("tab10", len(genes_clusters['cluster'].unique()))))
    row_colors = genes_clusters['cluster'].map(my_palette)
    row_colors.index = genes_clusters['Gene ID']
    
    
    # Create clustermap
    g = sns.clustermap(
        curves,
        method=cluster_method,
        figsize=figsize,
        metric=cluster_metric,
        col_cluster=col_cluster,  # Disable column clustering
        row_cluster=row_cluster,  # Disable row clustering
        yticklabels=yticklabels,  # Show all gene names as row labels
        xticklabels=xticklabels,  # Hide sample names
        z_score=z_score,
        cmap=cmap_color,
        dendrogram_ratio=dendrogram_ratio,
        cbar_pos=cbar_pos,
        row_colors=row_colors.loc[curves.index],
        annot_kws={"size": fontsize + 2}
    )
    
    
    
    reordered_indices = g.dendrogram_row.reordered_ind  # Get clustering order
    reordered_row_colors = row_colors.iloc[reordered_indices]  # Reorder row colors
    g.ax_row_colors.set_yticks(range(len(reordered_row_colors)))  # Align ticks with reordered rows
    

    # Explicitly set y-axis labels to gene names
    #g.ax_heatmap.set_yticks(range(len(curves.index)))
    #g.ax_heatmap.set_yticklabels(curves.index, fontsize=fontsize)

    # Update heatmap labels and appearance
    g.ax_heatmap.set_ylabel("Genes", fontsize=fontsize + 2)
    g.ax_heatmap.set_xlabel("Samples", fontsize=fontsize + 2)
    g.ax_heatmap.tick_params(axis='x', labelsize=fontsize)
    g.ax_heatmap.tick_params(axis='y', labelsize=fontsize)
    g.ax_cbar.tick_params(labelsize=fontsize)
    g.ax_row_colors.tick_params(labelsize=fontsize + 2)

    # Save heatmap as PDF
    g.savefig(save_path, format='pdf', dpi=300)
    print(f"Heatmap saved as {save_path}")
    
    
def get_noised_curves(adata: ad.AnnData = None,
                      cell_type: str = None,
                      filter_table_feature: str = 'R-squared',
                      filter_table_feature_pval: str = 'adjusted P-value',
                      table_filter_thr: float = 0.1,
                      table_filter_pval_thr: float = 0.05,
                      path_to_results: str = 'Results_PILOT/',df=None,name_col_fil='gene'):
    """
    Generates smoothed and noised gene expression curves based on fitted functions and pseudotime data.

    This function calculates gene expression curves by incorporating cell-level variance observed at each pseudotime point. 
    It scales the curves for downstream analysis and optionally filters genes based on statistical criteria.

    Parameters
    ----------
    adata : ad.AnnData, optional
        Annotated data matrix containing gene expression data (default is None).

    cell_type : str, optional
        The specific cell type to analyze. This identifies relevant cells and markers (default is None).

    filter_table_feature : str, optional
        Column name in the gene marker table used to filter genes based on a threshold (default is 'R-squared').

    filter_table_feature_pval : str, optional
        Column name in the gene marker table for filtering genes by p-value (default is 'adjusted P-value').

    table_filter_thr : float, optional
        Threshold for filtering genes based on the specified feature (default is 0.1).

    table_filter_pval_thr : float, optional
        P-value threshold for filtering genes (default is 0.05).

    path_to_results : str, optional
        Directory path to locate cell and marker data for the analysis (default is 'Results_PILOT/').

    df : pd.DataFrame, optional
        DataFrame containing additional gene metadata, including a column for gene identifiers (default is None).

    name_col_fil : str, optional
        Column name in `df` that matches gene identifiers with the 'Gene ID' column in the marker table (default is 'gene').

    Returns
    -------
    scaled_curves : pd.DataFrame
        Scaled smoothed expression curves for genes without added noise.

    scaled_noised_curves : pd.DataFrame
        Scaled smoothed expression curves for genes with added noise, derived from cell-level variance.

    pseudotime_sample_names : pd.DataFrame
        A mapping of pseudotime points to sample identifiers, including the first sample at each time point.

    Notes
    -----
    - The function assumes specific file structure and data organization within the `path_to_results` directory.
      - A subdirectory `/cells/` contains cell-specific data for each cell type.
      - A subdirectory `/Markers/` contains marker information for each cell type, including a file `Whole_expressions.csv`.
    - Noise is added to the gene expression curves based on variance at each pseudotime point and weighted by covariates from the marker table.
    - The output curves are scaled using `StandardScaler` to facilitate downstream analysis.

    """
    # Get cells of one cell_type 
    cells = pd.read_csv(path_to_results + "/cells/" + str(cell_type) + ".csv", index_col = 0)

    # Get the pseudotime points
    pseudotime_sample_names = cells[['sampleID', 'Time_score']].groupby('Time_score').first()
    pseudotime_sample_names = pseudotime_sample_names.sort_index()
    
    # Read the table of fitted function for each gene
    table = pd.read_csv(path_to_results + "/Markers/" + str(cell_type) + "/Whole_expressions.csv", index_col = 0)
    table = table.fillna(0)
    

    # Filter the table
    table=table[table['Gene ID'].isin(list(df[name_col_fil]))]
    #selected_table = table[ (np.abs(table[filter_table_feature]) >= table_filter_thr) & \
                            #(table[filter_table_feature_pval] <= table_filter_pval_thr) ]
    selected_table=table
    # Get curves of each gene
    curves = make_curves(selected_table, pseudotime_sample_names.index.values)

    # Compute cells standard deviation in each time point
    cells_genes = cells.loc[:, ~cells.columns.isin(['sampleID']) ]
    sample_genes_var = cells_genes.groupby('Time_score').std() / 10

    # Get the sum of the covariates for each gene
    sum_covariates = np.sum(list(selected_table[['Treat', 'Treat2', 'Intercept']].values * [1, 1, -1]), axis = 1)
    
    noise_values = sample_genes_var[selected_table['Gene ID']].mul(sum_covariates)
                                  
    noised_curves = curves + noise_values.transpose()                              
    noised_curves.columns = pseudotime_sample_names.sampleID

    noised_curves = noised_curves.fillna(0)
    
    scaler = StandardScaler()
    scaled_noised_curves = pd.DataFrame(scaler.fit_transform(noised_curves.transpose()).transpose())
    scaled_noised_curves.columns = noised_curves.columns
    scaled_noised_curves.index = noised_curves.index
    
    scaler = StandardScaler()
    scaled_curves = pd.DataFrame(scaler.fit_transform(curves.transpose()).transpose())
    scaled_curves.columns = curves.columns
    scaled_curves.index = curves.index
    
    return scaled_curves, scaled_noised_curves, pseudotime_sample_names
   


def go_enrichment_heatmap(adata, num_gos=10, fontsize=16, figsize=(15, 12), cmap='Blues', cmap_max=5, dpi=100, 
                          bbox_inches='tight', facecolor='white', transparent=False, organism='hsapiens', 
                          p_threshold=0.05, filter_go='skin',row_cluster=True,col_cluster=False, method='average',metric='euclidean',z_score=1,dendrogram_ratio=(0.2, 0.2),linewidths=0.5, path = 'Results_PILOT/GO/heatmaps/',convert_names=False,source_filter='GO:'):
    """
    Performs Gene Ontology (GO) enrichment analysis for clusters and generates a heatmap of enriched terms.

    This function uses the GProfiler API to perform GO enrichment analysis for each cluster of genes and visualizes 
    the most significantly enriched GO terms as a heatmap, with optional clustering of rows and/or columns.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing gene expression and cluster information.

    num_gos : int, optional
        Number of top enriched GO terms to display for each cluster. Default is 10.

    fontsize : int, optional
        Font size for the heatmap labels and annotations. Default is 16.

    figsize : tuple, optional
        Size of the figure (width, height) in inches. Default is (15, 12).

    cmap : str, optional
        Colormap to use for the heatmap. Default is 'Blues'.

    cmap_max : float, optional
        Maximum value for the colormap. Default is 5.

    dpi : int, optional
        Resolution of the saved plot in dots per inch. Default is 100.

    bbox_inches : str, optional
        Option for saving the plot with tight layout. Default is 'tight'.

    facecolor : str, optional
        Background color of the plot. Default is 'white'.

    transparent : bool, optional
        Whether the background of the saved plot is transparent. Default is False.

    organism : str, optional
        The organism used for GO enrichment analysis. Supported organisms include 'hsapiens' for humans. Default is 'hsapiens'.

    p_threshold : float, optional
        Threshold for filtering GO terms by p-value significance. Default is 0.05.

    filter_go : str, optional
        Keyword to exclude specific GO terms from the heatmap (e.g., 'skin'). Default is 'skin'.

    row_cluster : bool, optional
        Whether to cluster rows (GO terms) in the heatmap. Default is True.

    col_cluster : bool, optional
        Whether to cluster columns (clusters) in the heatmap. Default is False.

    method : str, optional
        Linkage method for clustering. Options include 'single', 'complete', 'average', etc. Default is 'average'.

    metric : str, optional
        Distance metric for clustering. Options include 'euclidean', 'correlation', etc. Default is 'euclidean'.

    z_score : int, optional
        If 0 or 1, standardizes rows or columns, respectively. Default is 1.

    dendrogram_ratio : tuple, optional
        Proportion of space used for row and column dendrograms in the heatmap. Default is (0.2, 0.2).

    linewidths : float, optional
        Width of the lines separating heatmap cells. Default is 0.5.

    path : str, optional
        Path to save the heatmap and intermediate results. Default is 'Results_PILOT/GO/heatmaps/'.

    convert_names : bool, optional
        Whether to convert Ensembl IDs to gene names using an external API. Default is False.
    
    source_filter : str, optional
        Filter for the 'source' column in GProfiler results. Only terms containing this value are included. Default is 'GO:'.

    Returns
    -------
    None
        The function saves the GO enrichment heatmap as a PNG file at the specified path.

    Notes
    -----
    - The function performs GO enrichment analysis for each cluster using GProfiler.
    - If the results for a cluster already exist as a CSV file, they are loaded to avoid redundant computations.
    - The heatmap displays -log10(p-value) of the enriched terms for each cluster, with optional clustering of terms and clusters.

    """
    if convert_names:
        genes=adata.uns['gene_selection_heatmap']['curves_activities'].copy()
        url = "https://biotools.fr/human/ensembl_symbol_converter/"
        ids = list(adata.uns['gene_selection_heatmap']['curves_activities'].index)

        # Convert the IDs list to JSON format
        ids_json = json.dumps(ids)

        # Create the payload for the POST request
        payload = {
            "api": 1,
            "ids": ids_json
        }

        # Send the POST request
        response = requests.post(url, data=payload)
        output = json.loads(response.text)
        output= pd.DataFrame(list(output.items()), columns=['Ensembl ID', 'Gene Name'])
        genes.reset_index(inplace=True)
        genes['Gene ID'] = genes['Gene ID'].map(output.set_index('Ensembl ID')['Gene Name'])
        genes.set_index('Gene ID', inplace=True)
        genes.reset_index(inplace=True)
        df=genes[['Gene ID','cluster']]
    else:
        genes=adata.uns['gene_selection_heatmap']['curves_activities'].copy()
        genes.reset_index(inplace=True)
        df=genes[['Gene ID','cluster']]
        
    os.makedirs(path, exist_ok=True) 
    gp = GProfiler(return_dataframe=True)
    clusters = df['cluster'].unique()
    heatmap_data = pd.DataFrame()

    for cluster in clusters:
        # Check if results already exist
        cluster_file = f"{path}GO_results_cluster_{cluster}.csv"
        if os.path.exists(cluster_file):
            print(f"Loading existing results for cluster {cluster}...")
            gprofiler_results = pd.read_csv(cluster_file)
        else:
            # Query GProfiler and save results
            cluster_genes = df[df['cluster'] == cluster]['Gene ID'].values
            gprofiler_results = gp.profile(organism=organism, query=list(cluster_genes))
            
            if not gprofiler_results.empty:
                gprofiler_results.to_csv(cluster_file, index=False)
            else:
                print(f"No GO terms found for cluster {cluster}. Skipping.")
                continue
            

        # Filter GO terms based on p-value threshold
        gprofiler_results = gprofiler_results[
            (gprofiler_results['p_value'] < p_threshold) &
            (gprofiler_results['source'].str.contains(source_filter, case=False))]
        if gprofiler_results.empty:
            print(f"No significant GO terms found for cluster {cluster}. Skipping.")
            continue

        # Select the top GO terms
        top_gos = gprofiler_results.head(num_gos)
        top_gos['cluster'] = cluster
        top_gos = top_gos[['name', 'p_value', 'cluster']]
        
        # Transform for heatmap
        top_gos['-log10(p_value)'] = -np.log10(top_gos['p_value'])
        heatmap_data = pd.concat([heatmap_data, top_gos])

    if heatmap_data.empty:
        print("No significant GO terms found across all clusters.")
        return

    # Pivot data for heatmap
    heatmap_data = heatmap_data[~heatmap_data['name'].str.contains(filter_go, case=False)]
    heatmap_pivot = heatmap_data.pivot_table(index='name', columns='cluster', values='-log10(p_value)', fill_value=0)

    # Create clustermap with dendrogram
    sns.clustermap(
        heatmap_pivot, cmap=cmap, vmax=cmap_max, figsize=figsize,row_cluster=row_cluster,col_cluster=col_cluster,method=method,metric=metric,z_score=z_score,
        cbar_kws={'label': '-log10(p_value)'}, dendrogram_ratio=dendrogram_ratio, linewidths=linewidths,
    )
    plt.title('GO Enrichment Heatmap with Clustering', fontsize=fontsize, y=1.05)
    plt.xlabel('Cluster', fontsize=fontsize)
    plt.ylabel('GO Terms', fontsize=fontsize)
    
    # Save heatmap
    plt.savefig(f"{path}GO_Enrichment_Heatmap.png", bbox_inches=bbox_inches, facecolor=facecolor, transparent=transparent)
    plt.show()

    
def results_gene_cluster_differentiation(cluster_name=None,sort_columns=['pvalue'],ascending=[True],threshold=0.5,p_value=0.01,converter=False):
    """
    Retrieve and sort gene cluster statistics based on specified criteria.

    This function filters and sorts gene cluster statistics based on fold change, p-value thresholds, and other criteria.
    Optionally, it converts Ensembl IDs to gene names using an external API.

    Parameters
    ----------
    cluster_name : str, optional
        The name of the gene cluster for which statistics are retrieved. Default is None.

    sort_columns : list of str, optional
        List of column names to sort the data by. Default is ['pvalue'].

    ascending : list of bool, optional
        List indicating the sorting order for each corresponding column. Default is [True].

    threshold : float, optional
        Minimum fold change (FC) value for selecting genes. Default is 0.5.

    p_value : float, optional
        Maximum p-value threshold for selecting genes. Default is 0.01.

    converter : bool, optional
        Whether to convert Ensembl IDs to gene names using the Biotools.fr API. Default is False.

    Returns
    -------
    pandas.DataFrame
        A sorted DataFrame containing gene cluster statistics filtered and sorted based on the specified criteria. 
        The returned DataFrame includes the following columns:
        - 'gene': Gene identifier or name (converted if `converter=True`).
        - 'cluster': Cluster name.
        - 'waldStat': Wald statistic for differential expression.
        - 'pvalue': P-value from the Wald test.
        - 'FC': Fold change.
        - 'Expression pattern': Pattern of gene expression.
        - 'fit-pvalue': P-value for model fit.
        - 'fit-mod-rsquared': Adjusted R-squared value for model fit.

    Notes
    -----
    - The function assumes that gene cluster statistics are stored in a CSV file named `gene_clusters_stats_extend.csv`
      located in the `Results_PILOT/` directory.
    - If `converter` is enabled, Ensembl IDs are converted to gene names using the Biotools.fr API. The API response is parsed
      and mapped to the gene column.
    """
   
    path='Results_PILOT/'
    statistics=pd.read_csv(path+'/gene_clusters_stats_extend.csv')
    df=statistics[statistics['cluster']==cluster_name]
    df['FC']=df['FC'].astype(float)
    df=df[df['FC'] > threshold]
    df['pvalue']=df['pvalue'].astype(float)
    df=df[df['pvalue'] < p_value]
    
   
    if converter:
        df['Ens_ID']=list(df['gene'])
        df_sorted = df.sort_values(by=sort_columns, ascending=ascending)
        genes=pd.read_csv('Results_PILOT/gene_clusters_stats_extend.csv')
        url = "https://biotools.fr/human/ensembl_symbol_converter/"
        ids = list(genes['gene'])
        ids_json = json.dumps(ids)
        payload = {
            "api": 1,
            "ids": ids_json
        }

        response = requests.post(url, data=payload)
        output = json.loads(response.text)
        output= pd.DataFrame(list(output.items()), columns=['Ensembl ID', 'Gene Name'])
        df_sorted['gene'] = df_sorted['gene'].map(output.set_index('Ensembl ID')['Gene Name'])
        return df_sorted[['gene','cluster','waldStat','pvalue','FC','Expression pattern','fit-pvalue','fit-mod-rsquared','Ens_ID']] 
    else:
        df_sorted = df.sort_values(by=sort_columns, ascending=ascending)
        return df_sorted[['gene','cluster','waldStat','pvalue','FC','Expression pattern','fit-pvalue','fit-mod-rsquared']] 
       