import numpy as np
import scipy
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from typing import Tuple
class Hermitian():
    
    def __init__(self, directed_net: nx.DiGraph):
        """
        Initializes the Hermitian object using a directed network, finds adjacency (A) and
        Hermitian adjacency (A_tilda) matrices for the directed network and normalizes the
        A_tilda usign Random-Walk method. 

        Parameters:
        - directed_net (nx.DiGraph): Directed network of lead-lag scores.
        """
        self.directed_net = directed_net
        
        # Calculate A, A_tilda and random-walk normalized A_tilda_rw
        self.a = self.calc_adjacency()
        self.a_tilda = self.calc_hermitian_adjacency()
        self.a_tilda_rw = self.norm_hermitian_adjacency()
    
    
    def calc_adjacency(self) -> scipy.sparse.csr_matrix:
        """
        Calculates the sparse adjacency matrix using a directed network.

        Returns:
        - scipy.sparse.csr_matrix: Adjacency matrix, A.
        """
        return nx.adjacency_matrix(self.directed_net)
    
    
    def calc_hermitian_adjacency(self) -> scipy.sparse.csr_matrix:
        """
        Calculates sparse Hermitian adjacency matrix, A_tilda, using the adjacency matrix A.

        Returns:
        - scipy.sparse.csr_matrix: Hermitian adjacency matrix, A_tilda.
        """
        return (self.a*1j) - (self.a.transpose()*1j)
    
    
    def norm_hermitian_adjacency(self) -> scipy.sparse.csr_matrix:
        """
        Normalize Hermitian adjacency matrix using Random Walk Normalization method.

        Returns:
        - scipy.sparse.csr_matrix: Random-Walk Normalized Hermitian adjacency matrix, A_tilda_rw.
        """
        # Step 1: Convert sparse adjacency matrix to array 
        adj_mat = self.a_tilda.toarray()

        # Step 2: Compute the degree matrix
        deg_mat = np.diag(np.sum(adj_mat, axis=1))

        # Step 3: Compute the inverse of the square root of the degree matrix
        inv_sqrt_deg_mat = np.linalg.inv(np.sqrt(deg_mat))

        # Step 4: Perform random walk normalization
        adj_mat_norm = np.dot(np.dot(inv_sqrt_deg_mat, adj_mat), inv_sqrt_deg_mat.T)

        # Step 4: Convert to sparse matrix
        adj_mat_norm = scipy.sparse.csr_matrix(adj_mat_norm)
        
        return adj_mat_norm
        
    
    def cluster_hermitian(self,
                          k: int,
                          kmeans_init: str = 'k-means++',
                          kmeans_n_init: int = 10,
                          kmeans_random_state: int = 42,
                          add_to_network = False) -> Tuple[np.ndarray, np.float64]:
        """
        Applies Hermitian clustering to the normalized Hermitian adjacency matrix for a given k.

        Parameters:
        - k (int): Number of clusters.
        - kmeans_init (str): Initialization method for KMeans. Default is 'k-means++'.
        - kmeans_n_init (int): Number of times KMeans will be run with different centroid seeds. Default is 10.
        - kmeans_random_state (int): Random state for KMeans. Default is 42.
        - add_to_network (bool): If True, adds cluster labels as an attribute to the nodes of directed neteork.

        Returns:
        - Tuple[np.ndarray, np.float64]: Returns a tuple (cluster_labels, silhouette_avg).
        """
        # find eigenvectors of Hermitian adjacency matrix
        eigenval, eigenvec = scipy.sparse.linalg.eigsh(self.a_tilda_rw, k=int(2 * math.floor(k / 2)))

        # Prepare input data for KMeans from eigenvectors
        X = np.block([[np.real(eigenvec), np.imag(eigenvec)]])

        # Apply Kmeans clustering
        clusterer = KMeans(
            n_clusters=k,
            init=kmeans_init,
            n_init=kmeans_n_init,
            random_state=kmeans_random_state
        )
        cluster_labels = clusterer.fit_predict(X)

        # Calculate Silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        # add cluster numbers to the network attributes
        if add_to_network:
            nx.set_node_attributes(self.directed_net, dict(zip(self.directed_net.nodes(), cluster_labels)), 'cluster')
              
        return cluster_labels, silhouette_avg
    
    
    def get_cluster_info(self) -> dict:
        """
        Retrieve cluster labels from the graph and generate a dictionary.

        Returns:
        - dict: Dictionary with cluster number as key and a list of node names as values.
        """
        cluster_labels = nx.get_node_attributes(self.directed_net, 'cluster')
        cluster_dict = {}
        for node, cluster in cluster_labels.items():
            if cluster in cluster_dict:
                cluster_dict[cluster].append(node)
            else:
                cluster_dict[cluster] = [node]
        return cluster_dict
    
    
    def cluster_hermitian_opt(self,
                              k_min: int,
                              k_max: int,
                              kmeans_init: str = 'k-means++',
                              kmeans_n_init: int = 10,
                              kmeans_random_state: int = 42) -> dict:
        """
        Applies Hermitian clustering to the normalized Hermitian adjacency matrix for a given range of k, between k_min and k_max.
        It uses silhouette score to find the optimal number of clusters between k_min and k_max.

        Parameters:
        - k_min (int): The minimum number of clusters to consider.
        - k_max (int): The maximum number of clusters to consider.
        - kmeans_init (str): Initialization method for KMeans. Default is 'k-means++'.
        - kmeans_n_init (int): Number of times KMeans will be run with different centroid seeds. Default is 10.
        - kmeans_random_state (int): Random state for KMeans. Default is 42.

        Returns:
        - dict: Returns a dictionary containing cluster number as keys and list of nodes in each cluster in directed network as values.
                It also adds the optimal cluster labels to the directed network of the class object.
        """
        # Initialize optimal number of clusters (ONC) clustering output
        onc_labels = None
        onc_score = -1

        # Do clustering in a given range of k_min and k_max and find the optimal k, in between
        for k in range(k_min, k_max + 1):
            labels, score = self.cluster_hermitian(k = k,
                                                   kmeans_init = kmeans_init,
                                                   kmeans_n_init = kmeans_n_init,
                                                   kmeans_random_state = kmeans_random_state)

            # Update optimal number of clusters, clustering outputs
            if score > onc_score:
                onc_score = score
                onc_labels = labels
                
        # Add cluster labels to the graph, if requested
        nx.set_node_attributes(self.directed_net, dict(zip(self.directed_net.nodes(), onc_labels)), 'cluster')
        return self.get_cluster_info()