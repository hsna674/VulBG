import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def calculate_distortion(data, k_range, sample_size=10000, batch_size=1000):
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[indices]
        print(f"Using sample of {sample_size} points for distortion calculation")
    else:
        sample_data = data
        print(f"Using all {len(data)} points for distortion calculation")
    
    sample_data = np.array(sample_data, dtype=np.float32)
    
    distortions = []
    silhouette_scores = []
    k_values = []
    
    print("Calculating distortion curve...")
    
    for k in tqdm(k_range, desc="Testing cluster numbers"):
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                batch_size=min(batch_size, len(sample_data)),
                max_iter=100,
                verbose=0
            )
            
            kmeans.fit(sample_data)
            
            distortion = kmeans.inertia_
            distortions.append(distortion)
            
            if k > 1 and k < min(50, len(sample_data)//2):
                try:
                    labels = kmeans.labels_
                    if len(np.unique(labels)) > 1:  # Ensure we have multiple clusters
                        sil_score = silhouette_score(sample_data, labels, sample_size=min(5000, len(sample_data)))
                        silhouette_scores.append(sil_score)
                    else:
                        silhouette_scores.append(0)
                except:
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
            
            k_values.append(k)
            
        except Exception as e:
            print(f"Error with k={k}: {e}")
            continue
    
    return k_values, distortions, silhouette_scores

def plot_distortion_curve(k_values, distortions, silhouette_scores, original_k=1140):
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1.plot(k_values, distortions, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Distortion (WCSS)', fontsize=12)
    ax1.set_title('Elbow Method - Distortion Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    if original_k in k_values:
        ax1.axvline(x=original_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'Original k={original_k}')
        ax1.legend()
    
    valid_sil_scores = [(k, s) for k, s in zip(k_values, silhouette_scores) if s > 0]
    if valid_sil_scores:
        k_sil, sil_vals = zip(*valid_sil_scores)
        ax2.plot(k_sil, sil_vals, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        best_k_idx = np.argmax(sil_vals)
        best_k = k_sil[best_k_idx]
        best_score = sil_vals[best_k_idx]
        ax2.scatter(best_k, best_score, color='red', s=100, zorder=5)
        ax2.annotate(f'Best: k={best_k}\nScore={best_score:.3f}', 
                    xy=(best_k, best_score), xytext=(10, 10),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    if len(distortions) > 2:
        rate_of_change = []
        for i in range(1, len(distortions)-1):
            rate = abs(distortions[i-1] - 2*distortions[i] + distortions[i+1])
            rate_of_change.append(rate)
        
        k_roc = k_values[1:-1]
        ax3.plot(k_roc, rate_of_change, 'mo-', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax3.set_ylabel('Rate of Change', fontsize=12)
        ax3.set_title('Rate of Change in Distortion', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        if rate_of_change:
            elbow_idx = np.argmax(rate_of_change)
            elbow_k = k_roc[elbow_idx]
            ax3.scatter(elbow_k, rate_of_change[elbow_idx], color='red', s=100, zorder=5)
            ax3.annotate(f'Potential Elbow: k={elbow_k}', 
                        xy=(elbow_k, rate_of_change[elbow_idx]), xytext=(10, 10),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    norm_distortions = [(d - min(distortions)) / (max(distortions) - min(distortions)) 
                       for d in distortions]
    
    ax4.plot(k_values, norm_distortions, 'b-', linewidth=2, label='Normalized Distortion')
    
    if valid_sil_scores:
        norm_silhouette = [1 - ((s - min(sil_vals)) / (max(sil_vals) - min(sil_vals))) 
                          if max(sil_vals) > min(sil_vals) else 0.5 for s in sil_vals]
        ax4.plot(k_sil, norm_silhouette, 'g-', linewidth=2, label='Inverted Norm. Silhouette')
    
    ax4.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax4.set_ylabel('Normalized Score', fontsize=12)
    ax4.set_title('Combined Metrics (Normalized)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('VulBG/kmeans_distortion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def recommend_k(k_values, distortions, silhouette_scores):
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING ANALYSIS RECOMMENDATIONS")
    print("="*60)
    
    if len(distortions) > 3:
        second_derivatives = []
        for i in range(1, len(distortions)-1):
            second_deriv = distortions[i-1] - 2*distortions[i] + distortions[i+1]
            second_derivatives.append(abs(second_deriv))
        
        if second_derivatives:
            elbow_idx = np.argmax(second_derivatives) + 1
            elbow_k = k_values[elbow_idx]
            print(f"📈 Elbow Method suggests: k = {elbow_k}")
    
    valid_sil_scores = [(k, s) for k, s in zip(k_values, silhouette_scores) if s > 0]
    if valid_sil_scores:
        k_sil, sil_vals = zip(*valid_sil_scores)
        best_sil_idx = np.argmax(sil_vals)
        best_sil_k = k_sil[best_sil_idx]
        best_sil_score = sil_vals[best_sil_idx]
        print(f"🎯 Silhouette Analysis suggests: k = {best_sil_k} (score: {best_sil_score:.3f})")
    print("="*60)

if __name__ == "__main__":
    print("Loading dataset...")
    with open("VulBG/dataset/processed/combined_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
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
    
    all_slice_embeddings = np.array(all_slice_embeddings, dtype=np.float32)
    print(f"Total slice embeddings: {len(all_slice_embeddings)}")
    print(f"Embedding dimension: {all_slice_embeddings.shape[1] if len(all_slice_embeddings) > 0 else 'N/A'}")
    
    k_range = list(range(2, 21)) + list(range(25, 101, 5)) + \
              list(range(100, 501, 25)) + list(range(500, 1501, 50))
    
    k_range = sorted(list(set(k_range)))
    print(f"Testing k values: {k_range[:5]}...{k_range[-5:]} ({len(k_range)} total)")
    
    k_values, distortions, silhouette_scores = calculate_distortion(
        all_slice_embeddings, 
        k_range, 
        sample_size=15000,
        batch_size=1000
    )
    
    fig = plot_distortion_curve(k_values, distortions, silhouette_scores, original_k=1140)
    
    recommend_k(k_values, distortions, silhouette_scores)
    
    results = {
        'k_values': k_values,
        'distortions': distortions,
        'silhouette_scores': silhouette_scores,
        'sample_size': len(all_slice_embeddings)
    }
    
    with open("VulBG/kmeans_analysis_results.pkl", "wb") as f:
        pickle.dump(results, f)