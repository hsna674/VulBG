import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class VulBGDataset(Dataset):
    def __init__(self, data, baseline_column="codebert"):
        self.data = data
        self.baseline_column = baseline_column
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "baseline": item.get(self.baseline_column, np.zeros(768)),
            "graph": item.get("graph_vec", np.zeros(128)),
            "label": item["vul"],
            "file": item["file"]
        }

def collate_fn(batch):
    baseline_inputs = torch.tensor([item["baseline"] for item in batch], dtype=torch.float32)
    graph_inputs = torch.tensor([item["graph"] for item in batch], dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    return baseline_inputs, labels, graph_inputs, len(batch)

class VulBGModel(nn.Module):
    def __init__(self, baseline_dim=768, graph_dim=128, hidden_dim=256, num_classes=2):
        super(VulBGModel, self).__init__()
        
        self.baseline_fc = nn.Linear(baseline_dim, hidden_dim)
        self.graph_fc = nn.Linear(graph_dim, hidden_dim)
        self.combined_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, baseline_input, graph_input):
        baseline_features = self.relu(self.baseline_fc(baseline_input))
        baseline_features = self.dropout(baseline_features)
        
        graph_features = self.relu(self.graph_fc(graph_input))
        graph_features = self.dropout(graph_features)
        
        combined = torch.cat([baseline_features, graph_features], dim=1)
        combined = self.relu(self.combined_fc(combined))
        combined = self.dropout(combined)
        
        output = self.classifier(combined)
        return output

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for baseline_inputs, labels, graph_inputs, batch_size in dataloader:
        baseline_inputs = baseline_inputs.to(device)
        labels = labels.to(device)
        graph_inputs = graph_inputs.to(device)
        
        optimizer.zero_grad()
        outputs = model(baseline_inputs, graph_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for baseline_inputs, labels, graph_inputs, batch_size in dataloader:
            baseline_inputs = baseline_inputs.to(device)
            labels = labels.to(device)
            graph_inputs = graph_inputs.to(device)
            
            outputs = model(baseline_inputs, graph_inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1

if __name__ == "__main__":
    with open("VulBG/dataset/processed/combined_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    print(f"Loaded dataset with {len(dataset)} samples")
    
    valid_dataset = []
    for item in dataset:
        if "codebert" in item and "graph_vec" in item:
            valid_dataset.append(item)
    
    print(f"Valid samples: {len(valid_dataset)}")
    
    train_data, test_data = train_test_split(valid_dataset, test_size=0.2, random_state=42, 
                                           stratify=[item["vul"] for item in valid_dataset])
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42,
                                          stratify=[item["vul"] for item in train_data])
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    train_dataset = VulBGDataset(train_data)
    val_dataset = VulBGDataset(val_data)
    test_dataset = VulBGDataset(test_data)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VulBGModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device}")
    
    num_epochs = 20
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "VulBG/best_model.pth")
            print(f"New best model saved! F1: {val_f1:.4f}")
    
    model.load_state_dict(torch.load("VulBG/best_model.pth"))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device
    )
    
    print("\nFinal Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")