import pickle
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset():
    with open("VulBG/dataset/processed/combined_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    total_samples = len(dataset)
    vul_samples = sum(1 for item in dataset if item["vul"] == 1)
    novul_samples = total_samples - vul_samples
    
    dataset_counts = Counter(item.get("dataset", "unknown") for item in dataset)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.pie([vul_samples, novul_samples], labels=['Vulnerable', 'Non-vulnerable'], 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Vulnerability Distribution')
    
    datasets = list(dataset_counts.keys())
    counts = list(dataset_counts.values())
    ax2.bar(datasets, counts)
    ax2.set_title('Samples per Dataset')
    ax2.set_ylabel('Number of Samples')
    
    code_lengths = [len(item["code"]) for item in dataset[:1000]]
    ax3.hist(code_lengths, bins=50, alpha=0.7)
    ax3.set_title('Code Length Distribution (Sample)')
    ax3.set_xlabel('Code Length (characters)')
    ax3.set_ylabel('Frequency')
    
    slice_counts = [len(item.get("slices", [])) for item in dataset]
    ax4.hist(slice_counts, bins=20, alpha=0.7)
    ax4.set_title('Number of Slices per File')
    ax4.set_xlabel('Number of Slices')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('VulBG/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Dataset Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Vulnerable: {vul_samples} ({vul_samples/total_samples*100:.1f}%)")
    print(f"Non-vulnerable: {novul_samples} ({novul_samples/total_samples*100:.1f}%)")
    print(f"Dataset distribution: {dict(dataset_counts)}")

if __name__ == "__main__":
    analyze_dataset()