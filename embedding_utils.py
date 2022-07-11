# code from: https://github.com/SKvtun/ParametricUMAP-Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample
from umap.umap_ import find_ab_params
import numpy as np
from IPython.core.debugger import set_trace


def umap_criterion_compatibility(criterion):
    
    '''
    Model output is a list of activations, 
    but we need only last item in list for cal
    '''
    
    def wrapper(*args, **kwargs):
        return criterion(args[0][-1].T)
    return wrapper

class UMAPDataset(Dataset):

    def __init__(self, data, epochs_per_sample, head, tail, weight, device='cpu', batch_size=1000,
                 use_epoch_per_sample=True):

        """
        create dataset for iteration on graph edges
        """
        self.weigh = weight
        self.batch_size = batch_size # number of POINTS in one batch
        self.data = data
        self.device = device
        self.use_epoch_per_sample = use_epoch_per_sample
        
        assert batch_size%2 == 0
        assert batch_size > 1
        assert len(head) == len(tail)
        
        if epochs_per_sample is None:
            epochs_per_sample = np.ones_like(head)
        
        self.edge_batch_size = batch_size // 2 # one edge per TWO points

        # repeat according to `epochs_per_sample` rate
        if self.use_epoch_per_sample:
            self.edges_to_exp = np.repeat(head, epochs_per_sample.astype("int"))
            self.edges_from_exp = np.repeat(tail, epochs_per_sample.astype("int"))
        else:
            self.edges_to_exp = head
            self.edges_from_exp = tail

        assert len(self.edges_to_exp) == len(self.edges_from_exp)
        
        self.num_edges = len(self.edges_to_exp)

        # shuffle edges
        shuffle_mask = np.random.permutation(range(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask]
        self.edges_from_exp = self.edges_from_exp[shuffle_mask]
        
        # was `int(self.num_edges / self.batch_size / 5)`
        self.batches_per_epoch = int(self.num_edges // self.batch_size)
        
    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        
        assert idx < len(self)
        
        # indexes to choose random batch
#         rand_index = np.random.randint(0, len(self.edges_to_exp) - 1, size=self.edge_batch_size)

        batch_index_to = self.edges_to_exp[idx:idx+self.edge_batch_size]
        batch_index_from = self.edges_from_exp[idx:idx+self.edge_batch_size]

        batch_to = torch.Tensor(self.data[batch_index_to]).to(self.device)
        batch_from = torch.Tensor(self.data[batch_index_from]).to(self.device)

        batch = torch.cat([batch_to, batch_from], dim=0)

        # concatenated (batch_to, batch_from)
        return batch
            
    def __len__(self):
        return self.batches_per_epoch
            
            

class ConstructUMAPGraph:

    def __init__(self, metric='euclidean', n_neighbors=10, random_state=42):
        
        self.random_state=random_state
        self.metric=metric # distance metric
        self.n_neighbors=n_neighbors # number of neighbors for computing k-neighbor graph

    @staticmethod
    def get_graph_elements(graph_, n_epochs):

        """
        gets elements of graphs, weights, and number of epochs per edge
        Parameters
        ----------
        graph_ : scipy.sparse.csr.csr_matrix
            umap graph of probabilities
        n_epochs : int
            maximum number of epochs per edge
        Returns
        -------
        graph scipy.sparse.csr.csr_matrix
            umap graph
        epochs_per_sample np.array
            number of epochs to train each sample for
        head np.array
            edge head
        tail np.array
            edge tail
        weight np.array
            edge weight
        n_vertices int
            number of verticies in graph
        """

        graph = graph_.tocoo()
        # eliminate duplicate entries by summing them together
        graph.sum_duplicates()
        # number of vertices in dataset
        n_vertices = graph.shape[1]
        
        # get the number of epochs based on the size of the dataset
        if n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] < 10000:
                n_epochs = 500
            else:
                n_epochs = 200
        
        # remove elements with very low probability
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        
        graph.eliminate_zeros()
        
        # get epochs per sample based upon edge probability
        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col
        weight = graph.data

        return graph, epochs_per_sample, head, tail, weight, n_vertices

    def __call__(self, X):
        
        # number of trees in random projection forest
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        # get nearest neighbors
        nnd = NNDescent(
            X.reshape((len(X), np.product(np.shape(X)[1:]))),
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        
        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph

        # build fuzzy_simplicial_set
        umap_graph, _, _ = fuzzy_simplicial_set(
            X=X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )

        graph, epochs_per_sample, head, tail, weight, n_vertices = self.get_graph_elements(umap_graph, None)
        
        self.graph = graph
        
        return epochs_per_sample, head, tail, weight


class UMAPLoss(nn.Module):

    def __init__(self, 
                 device='cpu', 
                 min_dist=0.1, 
                 batch_size=1000, 
                 negative_sample_rate=5,
                 edge_weight=None, 
                 repulsion_strength=1.0):

        """
        batch_size : int
        size of mini-batches
        negative_sample_rate : int
          number of negative samples PER  each positive sample to train on
        _a : float
          distance parameter in embedding space
        _b : float float
          distance parameter in embedding space
        edge_weights : array
          weights of all edges from sparse UMAP graph
        repulsion_strength : float, optional
          strength of repulsion vs attraction for cross-entropy, by default 1.0
        """

        super().__init__()
        self.device = device
        self._a, self._b = find_ab_params(1.0, min_dist)
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength

    @staticmethod
    def convert_distance_to_probability(distances, a=1.0, b=1.0):
        return 1.0 / (1.0 + a * distances ** (2. * b))

    def compute_cross_entropy(self, 
                              probabilities_graph, # GT graph
                              probabilities_distance, # predicted probs
                              EPS=1e-4, 
                              repulsion_strength=1.0):
        # cross entropy
        attraction_term = -probabilities_graph * torch.log(
            torch.clamp(probabilities_distance, EPS, 1.0)
        )

        repellant_term = -(1.0 - probabilities_graph) * torch.log(torch.clamp(
            1.0 - probabilities_distance, EPS, 1.0
        )) * self.repulsion_strength
        CE = attraction_term + repellant_term
        return attraction_term, repellant_term, CE

    def forward(self, embedding, *args, **kwargs):
        
        
        '''
        embedding - stacked pair of [embedding_to, embedding_from]
        embedding_to - embedding points that correspond to "heads"
        embedding_from - embedding points that correspond to "tails"
        '''
        
        N_samples = embedding.shape[0]
        batch_size = N_samples//2

        embedding_to, embedding_from = embedding[:N_samples//2,...], embedding[N_samples//2:,...]
        
        assert embedding_to.shape == embedding_from.shape
        if batch_size < 1e+3:
            assert self.negative_sample_rate == 0, "Negative samples makes no sense for low batch size!"
        
        device = self.device
        
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, 
                                                   self.negative_sample_rate, 
                                                   dim=0)
        
        embedding_neg_from = torch.repeat_interleave(embedding_from, 
                                                     self.negative_sample_rate, 
                                                     dim=0)
        
        # random permutation of `embedding_neg_from`
        embedding_neg_from = torch.index_select(embedding_neg_from, 
                                                0, 
                                                torch.randperm(embedding_neg_from.size(0), device=device))

        #  distances between samples (and negative samples)
        distance_embedding = torch.cat(
            [
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
            ],
            dim=0)

        # convert probabilities to distances
        probabilities_distance = self.convert_distance_to_probability(
            distance_embedding, self._a, self._b
        )

        # set true probabilities based on negative sampling
        # WHY ITS BINARY
        probabilities_graph = torch.cat(
            [
             torch.ones(batch_size, device=device), 
             torch.zeros(batch_size * self.negative_sample_rate, device=device)
            ],
            dim=0
        )
        
        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = self.compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self.repulsion_strength,
        )

        return torch.mean(ce_loss)