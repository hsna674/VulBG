import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sys
from node2vec import Node2Vec
import networkx as nx

def train_kmeans(n_clusters, inputs, random_state=0):
    print(f"Training K-means with {n_clusters} clusters on {len(inputs)} samples...")
    
    inputs = np.array(inputs, dtype=np.float32)
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        verbose=1, 
        max_iter=300, 
        batch_size=min(3000, len(inputs))
    )
    
    for i in range(0, len(inputs), 3000):
        batch = inputs[i:i+3000]
        batch = np.array(batch, dtype=np.float32)
        kmeans.partial_fit(batch)
    
    return kmeans

def generate_edges(dataset, cluster_model, n_clusters, use_weights=False):
    edges = []
    func_idx_begin = n_clusters
    
    print("Generating edges...")
    for i, func in enumerate(dataset):
        if "slices_vec" not in func or len(func["slices_vec"]) == 0:
            continue
            
        slice_embeddings = func["slices_vec"]
        if isinstance(slice_embeddings, list):
            slice_embeddings = np.array(slice_embeddings, dtype=np.float32)
        else:
            slice_embeddings = slice_embeddings.astype(np.float32)
        
        slice_embeddings = slice_embeddings * 1000.0
        
        slice_embeddings = slice_embeddings.astype(np.float32)
        
        try:
            cluster_labels = cluster_model.predict(slice_embeddings)
        except Exception as e:
            print(f"Error predicting clusters for function {i}: {e}")
            print(f"Slice embeddings shape: {slice_embeddings.shape}")
            print(f"Slice embeddings dtype: {slice_embeddings.dtype}")
            continue
        
        func_node = func_idx_begin + i
        
        for j, cluster_label in enumerate(cluster_labels):
            if use_weights:
                distance = np.linalg.norm(
                    cluster_model.cluster_centers_[cluster_label] - slice_embeddings[j]
                )
                weight = 10000 / (distance + 1)
                edges.append((func_node, cluster_label, weight))
            else:
                edges.append((func_node, cluster_label))
    
    return edges

def create_networkx_graph(edges, use_weights=False):
    G = nx.Graph()
    
    if use_weights:
        for source, target, weight in edges:
            G.add_edge(source, target, weight=weight)
    else:
        G.add_edges_from(edges)
    
    return G

def generate_node_embeddings(G, dataset, n_clusters, embedding_dim=128):
    print("Generating node embeddings with Node2Vec...")
    
    if G.number_of_nodes() == 0:
        print("Warning: Graph has no nodes!")
        return dataset
    
    node2vec = Node2Vec(
        G, 
        dimensions=embedding_dim,
        walk_length=30,
        num_walks=200,
        workers=4
    )
    
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    func_idx_begin = n_clusters
    
    for i, func in enumerate(dataset):
        func_node = func_idx_begin + i
        
        if str(func_node) in model.wv:
            func["graph_vec"] = model.wv[str(func_node)].astype(np.float32)
        else:
            func["graph_vec"] = np.zeros(embedding_dim, dtype=np.float32)
            print(f"Warning: Node {func_node} not found in graph")
    
    return dataset

if __name__ == "__main__":
    N_CLUSTERS = 1140
    USE_WEIGHTS = False
    EMBEDDING_DIM = 128
    
    dataset_path = "VulBG/dataset/processed/combined_dataset.pkl"
    print("Loading dataset...")
    
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    print(f"Dataset size: {len(dataset)}")
    
    print("Collecting slice embeddings...")
    all_slice_embeddings = []
    
    for func in dataset:
        if "slices_vec" in func and len(func["slices_vec"]) > 0:
            slice_embeddings = func["slices_vec"]
            if isinstance(slice_embeddings, list):
                slice_embeddings = np.array(slice_embeddings, dtype=np.float32)
            else:
                slice_embeddings = slice_embeddings.astype(np.float32)
            
            slice_embeddings = slice_embeddings * 1000.0
            slice_embeddings = slice_embeddings.astype(np.float32)
            
            all_slice_embeddings.extend(slice_embeddings)
    
    print(f"Total slice embeddings: {len(all_slice_embeddings)}")
    
    all_slice_embeddings = np.array(all_slice_embeddings, dtype=np.float32)
    
    kmeans = train_kmeans(N_CLUSTERS, all_slice_embeddings)
    
    edges = generate_edges(dataset, kmeans, N_CLUSTERS, USE_WEIGHTS)
    print(f"Generated {len(edges)} edges")
    
    if len(edges) == 0:
        print("Warning: No edges generated! This might indicate a problem with the data.")
        sys.exit(1)
    
    G = create_networkx_graph(edges, USE_WEIGHTS)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    dataset = generate_node_embeddings(G, dataset, N_CLUSTERS, EMBEDDING_DIM)
    
    print("Saving final dataset...")
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    
    with open("VulBG/dataset/processed/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    print("Behavior graph generation completed!")