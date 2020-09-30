import numpy as np

class PercolateNetwork:
    def __init__(self, adj_matrix, method='random', ranking=None):
        """
        :params adj_matrix: A_{i,j} = 1 describes a directed edge from i -> j
        :param percolate_by_degree: If true the order in which vertices are added will be in ascending degree order
        :return S(phi): Cluster size (as proportion of the network) function of phi
        """
        self.n_verts = adj_matrix.shape[0]
        self.largest_cluster_size = [] # Record appended after each step
        self.adj_matrix = adj_matrix
        self.method = method
        
        if type(ranking) == type(None):
            self.ranking = np.zeros((adj_matrix.shape[0],1))
        else:
            self.ranking = ranking.reshape((-1,1))
        
        
    def init_percolate(self):
        self.num_clusters = 0
        self.clusters = {} # Dict of cluster id's with values lists of vertex members
        self.vertex_cluster_map = {} # Key (Vertex id) Value (Cluster Number)
        self.adj_matrix, self.ranking = shuffle(self.adj_matrix, self.ranking) # Introduce randomness to ordering over iterations safely
        
        # Choose ordering in which vertices are added
        if self.method == 'degree':
            # Sort vertices indices by ascending degree
            # Introduce randomness to order of sort for equal degrees
            # Degrees will be removed in desc order 
            self.degrees = self.adj_matrix.sum(axis=1)
            self.vertex_ordering = list(np.argsort(self.degrees))
        elif self.method == 'random':
            # Random ordering of addition
            self.vertex_ordering = list(np.arange(0,self.n_verts))
        elif self.method == 'ranking':
            # Custom ranking criteria (eg. PageRank)
            self.vertex_ordering = list(np.argsort(self.ranking.reshape((-1,))))
    
    def percolate(self, n_iters=1000, verbose=False):
        for iter_t in range(n_iters):
            if verbose: 
                print(iter_t)
                
            self.init_percolate()
            self.largest_cluster_size.append([]) # New iteration of cluster sizes
            
            while len(self.vertex_ordering) > 0:
                if verbose and len(self.vertex_ordering) % 200 == 0: 
                    print(len(self.vertex_ordering))
                    
                # Create a new cluster of a single vertex
                next_vert = self.vertex_ordering.pop(0)
                self.num_clusters += 1
                self.clusters[self.num_clusters] = [next_vert]
                self.vertex_cluster_map[next_vert] = self.num_clusters
                
                # Iterate over edges and add edges where the vertex already exists in the network
                vert_edges = np.nonzero(self.adj_matrix[next_vert, :])[0]
                neighbouring_verts_in_network = list(filter(lambda x: x in self.vertex_cluster_map, vert_edges))

                for other_vert in neighbouring_verts_in_network:
                    # Join clusters if different
                    vert_cluster_idx = self.vertex_cluster_map[next_vert]
                    other_cluster_idx = self.vertex_cluster_map[other_vert]

                    if vert_cluster_idx != other_cluster_idx:
                        vert_cluster = self.clusters[vert_cluster_idx] 
                        other_cluster = self.clusters[other_cluster_idx]

                        # Expand other_cluster
                        self.clusters[other_cluster_idx] = other_cluster + vert_cluster

                        # Relabel Verts
                        for relabel_vert in vert_cluster:
                            self.vertex_cluster_map[relabel_vert] = other_cluster_idx

                # Update largest cluster
                cluster_size = 0
                for cluster_idx, verts_in_cluster in self.clusters.items():
                    size = len(verts_in_cluster)

                    if size > cluster_size:
                        cluster_size = size

                self.largest_cluster_size[iter_t].append(cluster_size)
                
            if verbose:
                print('iter_finished')
                
        # Average Cluster Sizes
        avg_cluster_sizes = np.array(self.largest_cluster_size).mean(axis=0)
        normalised_avg_cluster_sizes = avg_cluster_sizes / self.n_verts
        
        # We must include the trivial zero vertices added for correct calculation
        normalised_avg_cluster_sizes = np.array([0] + list(normalised_avg_cluster_sizes))
        
        # Compute Cluster size as function of phi
        # In notation of:
        # Newman, Mark. Networks: An Introduction (Kindle Location 11509). Oxford University Press, USA. Kindle Edition. 
        
        def S(phi):
            # Edge cases
            if phi == 0:
                return 0
            if phi == 1:
                return normalised_avg_cluster_sizes[self.n_verts - 1]
            
            n = self.n_verts
            p = phi
            
            def vectorised_pmf(ks):
                ks_end = n - ks + 1

                X = np.repeat(np.arange(0,n+1).reshape(1,-1),repeats=ks.shape[0], axis=0)
                Y = np.repeat(np.arange(0,n+1).reshape(1,-1),repeats=ks.shape[0], axis=0)


                for row in range(X.shape[0]):
                    X[row,:ks[row] + 1] = 1 # When we take log this is zero
                    Y[row, ks_end[row]:] = 1

                Y[:, 0] = 1 # Fix up for log(0)

                combination_num = np.log(X).sum(axis=1)
                combination_den = np.log(Y).sum(axis=1)
                combination_log = combination_num - combination_den

                p_k_log = ks * np.log(p)
                neg_p_K_log = (n - ks) * np.log(1 - p) # (1-p)^(n-k)
                p_log = combination_log + p_k_log + neg_p_K_log
                probability = np.exp(np.maximum(p_log, np.log(10e-15))) # Prevent overflow

                return probability
            
            r = np.arange(0, normalised_avg_cluster_sizes.shape[0])
            probabilities = vectorised_pmf(r)

            s_phi = (probabilities * normalised_avg_cluster_sizes).sum()
            return s_phi
        
        return S