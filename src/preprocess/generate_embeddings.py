import torch
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
import pickle
import gc
from tqdm import tqdm
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model = model.to(device)
model.eval()

def embed_code_batch(codes, max_length=512):
    inputs = tokenizer(
        codes, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=max_length
    )
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    
    return embeddings

def embed_slices_batch(slices_list, max_length=256):
    all_slices = []
    slice_counts = []
    
    for slices in slices_list:
        all_slices.extend(slices)
        slice_counts.append(len(slices))
    
    if not all_slices:
        return []
    
    slice_embeddings = embed_code_batch(all_slices, max_length)
    
    result = []
    start_idx = 0
    for count in slice_counts:
        end_idx = start_idx + count
        result.append(slice_embeddings[start_idx:end_idx])
        start_idx = end_idx
    
    return result

def collate_fn(batch):
    codes = [item["code"] for item in batch]
    slices = [item["slices"] for item in batch]
    files = [item["file"] for item in batch]
    labels = [item["vul"] for item in batch]
    
    return codes, slices, files, labels

if __name__ == "__main__":
    dataset_path = "VulBG/dataset/processed/combined_dataset.pkl"
    print("Loading dataset...")
    
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    print(f"Dataset size: {len(dataset)}")
    
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=False)
    
    print("Generating embeddings...")
    
    for i, (codes, slices_list, files, labels) in enumerate(tqdm(dataloader)):
        code_embeddings = embed_code_batch(codes)
        
        slice_embeddings = embed_slices_batch(slices_list)
        
        for j, file in enumerate(files):
            dataset_idx = i * batch_size + j
            dataset[dataset_idx]["codebert"] = code_embeddings[j]
            if j < len(slice_embeddings):
                dataset[dataset_idx]["slices_vec"] = slice_embeddings[j]
            else:
                dataset[dataset_idx]["slices_vec"] = code_embeddings[j:j+1]
        
        if i % 10 == 0:
            gc.collect()
            torch.mps.empty_cache()
    
    print("Saving embeddings...")
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    
    print("Embeddings generated successfully!")