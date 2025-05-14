import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import json
import numpy as np
from collections import defaultdict, Counter
import random
from model import DualEncoder

SEED = 42
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {"GPT4":0,"Mistral":1,"Sonnet":2,"Gemini":3,"Phi3":4,"Novice":5,"Llama31405B":6,"Llama318B":7,"Expert":8}
reverse_label_map = {v: k for k, v in label_map.items()}

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MODEL_NAME = "deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EnsembleModel:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model = DualEncoder(MODEL_NAME, len(label_map)).to(DEVICE)
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)
    
    def predict(self, history_ids, history_mask, response_ids, response_mask):
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(history_ids, history_mask, response_ids, response_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        # 投票机制
        vote_counts = Counter(all_preds)
        max_count = max(vote_counts.values())
        candidates = [k for k, v in vote_counts.items() if v == max_count]
        
        # 平票时随机选择一个
        return random.choice(candidates)

def process_test_data(model_paths):
    # 加载集成模型
    ensemble = EnsembleModel(model_paths)
    
    # 加载测试数据
    with open("mrbench_v3_testset.json") as f:
        data = json.load(f)
    
    results = []
    
    # 处理每个对话
    for item in data:
        sample = {
            "conversation_id": item["conversation_id"],
            "conversation_history": item["conversation_history"],
            "tutor_responses": {}
        }
        
        # 处理每个tutor的回复
        for tutor, resp in item["tutor_responses"].items():
            response = resp["response"]
            
            # Tokenize输入
            history_enc = tokenizer(
                item["conversation_history"], 
                truncation=True, 
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt"
            ).to(DEVICE)
            
            response_enc = tokenizer(
                response,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt"
            ).to(DEVICE)
            
            # 获取集成预测结果
            pred = ensemble.predict(
                history_enc["input_ids"],
                history_enc["attention_mask"],
                response_enc["input_ids"],
                response_enc["attention_mask"]
            )
            
            # 保存结果
            sample["tutor_responses"][tutor] = {
                "response": response,
                "annotation": {
                    "Tutor_Identification": reverse_label_map[pred]
                }
            }
        
        results.append(sample)
    
    # 保存最终结果
    with open("ensemble_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    model_paths = [f"best_model_fold{i}.pt" for i in range(1,6)]
    process_test_data(model_paths)