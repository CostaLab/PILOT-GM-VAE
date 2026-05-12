__version__ = "0.1.1"
__author__ = 'Mehdi Joodaki'
__credits__ = 'Institute for Computational Genomics'

from .core import (train_gmvae, plot_umap_and_stacked_bar, gmmvae_wasserstein_distance,
                   gaussian_mixture_vae_representation, filter_cells_by_sample_and_cell_type,
                   genes_selection_heatmap, go_enrichment_heatmap, results_gene_cluster_differentiation)
from .model import GMVAE
from .networks import GMVAENet, InferenceNet, GenerativeNet, Flatten, Reshape, GumbelSoftmax, Gaussian
from .losses import LossFunctions
from .metrics import Metrics