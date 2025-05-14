import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import json
import numpy as np
from model import DualEncoder

# 配置参数
SEED = 42
BATCH_SIZE = 8
MAX_LEN = 512
EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {"GPT4":0,"Mistral":1,"Sonnet":2,"Gemini":3,"Phi3":4,"Novice":5,"Llama31405B":6,"Llama318B":7,"Expert":8}
MODEL_NAME = "deberta-v3-base"
N_FOLDS = 5

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# 加载数据
def load_train_data(filename):
    with open(filename) as f:
        data = json.load(f)
    samples = []
    for item in data:
        history = item["conversation_history"]
        for tutor, resp in item["tutor_responses"].items():
            samples.append({
                "history": history,
                "response": resp["response"],
                "label": label_map[tutor],
                "annotations": resp["annotation"]
            })
    return samples

# 数据集类
class EnhancedDataset(Dataset):
    def __init__(self, samples, label_map):
        self.samples = samples
        self.label_map = label_map
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample

# 数据预处理
def collate_fn(batch):
    histories = [tokenizer(
        item["history"], 
        truncation=True, 
        max_length=MAX_LEN,
        padding="max_length"
    ) for item in batch]
    
    responses = [tokenizer(
        item["response"], 
        truncation=True, 
        max_length=MAX_LEN,
        padding="max_length"
    ) for item in batch]
    
    return {
        "history_ids": torch.stack([torch.tensor(h["input_ids"]) for h in histories]),
        "history_mask": torch.stack([torch.tensor(h["attention_mask"]) for h in histories]),
        "response_ids": torch.stack([torch.tensor(r["input_ids"]) for r in responses]),
        "response_mask": torch.stack([torch.tensor(r["attention_mask"]) for r in responses]),
        "labels": torch.tensor([item["label"] for item in batch])
    }

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 训练函数
def train_epoch(model, train_loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(
                batch["history_ids"].to(DEVICE),
                batch["history_mask"].to(DEVICE),
                batch["response_ids"].to(DEVICE),
                batch["response_mask"].to(DEVICE)
            )
            loss = criterion(outputs, batch["labels"].to(DEVICE))
        
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].cpu().numpy())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 评估函数
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["history_ids"].to(DEVICE),
                batch["history_mask"].to(DEVICE),
                batch["response_ids"].to(DEVICE),
                batch["response_mask"].to(DEVICE)
            )
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
    
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted")
    }

# 主流程
def main():
    data = load_train_data("mrbench_v3_devset.json")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        
        # 创建数据集
        train_dataset = Subset(EnhancedDataset(data, label_map), train_idx)
        val_dataset = Subset(EnhancedDataset(data, label_map), val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn)
        
        # 初始化模型
        model = DualEncoder(MODEL_NAME, len(label_map)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        
        best_f1 = 0
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_metrics = evaluate(model, val_loader)
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")
                print(f"New best model saved with F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()