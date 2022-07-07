import torch
import torch.nn as nn
from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample
from umap.umap_ import find_ab_params
import numpy as np

class UMAPDataset:

    def __init__(self, data, epochs_per_sample, head, tail, weight, device='cpu', batch_size=1000):

        """
        create dataset for iteration on graph edges
        """
        self.weigh = weight
        self.batch_size = batch_size
        self.data = data
        self.device = device

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        self.num_edges = len(self.edges_to_exp)

        # shuffle edges
        shuffle_mask = np.random.permutation(range(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask]
        self.edges_from_exp = self.edges_from_exp[shuffle_mask]

    def get_batches(self):
        batches_per_epoch = int(self.num_edges / self.batch_size / 5)
        for _ in range(batches_per_epoch):
            rand_index = np.random.randint(0, len(self.edges_to_exp) - 1, size=self.batch_size)
            batch_index_to = self.edges_to_exp[rand_index]
            batch_index_from = self.edges_from_exp[rand_index]
            if self.device == 'cuda':
                batch_to = torch.Tensor(self.data[batch_index_to]).cuda()
                batch_from = torch.Tensor(self.data[batch_index_from]).cuda()
            else:
                batch_to = torch.Tensor(self.data[batch_index_to])
                batch_from = torch.Tensor(self.data[batch_index_from])
            yield (batch_to, batch_from)
            
            
            
import torch
import torch.nn as nn
from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample
from umap.umap_ import find_ab_params
import numpy as np


class ConstructUMAPGraph:

    def __init__(self, metric='euclidean', n_neighbors=10, batch_size=1000, random_state=42):
        self.batch_size=batch_size
        self.random_state=random_state
        self.metric=metric # distance metric
        self.n_neighbors=n_neighbors # number of neighbors for computing k-neighbor graph

        pass

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
            if graph.shape[0] <= 10000:
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
        umap_graph, sigmas, rhos = fuzzy_simplicial_set(
            X=X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )

        graph, epochs_per_sample, head, tail, weight, n_vertices = self.get_graph_elements(umap_graph, None)
        return epochs_per_sample, head, tail, weight


class UMAPLoss(nn.Module):

    def __init__(self, device='cpu', min_dist=0.1, batch_size=1000, negative_sample_rate=5,
                 edge_weight=None, repulsion_strength=1.0):

        """
        batch_size : int
        size of mini-batches
        negative_sample_rate : int
          number of negative samples per positive samples to train on
        _a : float
          distance parameter in embedding space
        _b : float float
          distance parameter in embedding space
        edge_weights : array
          weights of all edges from sparse UMAP graph
        parametric_embedding : bool
          whether the embeddding is parametric or nonparametric
        repulsion_strength : float, optional
          strength of repulsion vs attraction for cross-entropy, by default 1.0
        """

        super().__init__()
        self.device = device
        self._a, self._b = find_ab_params(1.0, min_dist)
        self.batch_size = batch_size
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength

    @staticmethod
    def convert_distance_to_probability(distances, a=1.0, b=1.0):
        return 1.0 / (1.0 + a * distances ** (2. * b))

    def compute_cross_entropy(self, probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0):
        # cross entropy
        attraction_term = -probabilities_graph * torch.log(
            torch.clamp(probabilities_distance, EPS, 1.0)
        )

        repellant_term = -(1.0 - probabilities_graph) * torch.log(torch.clamp(
            1.0 - probabilities_distance, EPS, 1.0
        )) * self.repulsion_strength
        CE = attraction_term + repellant_term
        return attraction_term, repellant_term, CE

    def forward(self, embedding_to, embedding_from):
        
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
        probabilities_graph = torch.cat(
            [
             torch.ones(self.batch_size, device=device), 
             torch.zeros(self.batch_size * self.negative_sample_rate, device=device)
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