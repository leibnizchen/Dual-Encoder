import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
class DualEncoder(nn.Module): # {'accuracy': 0.8951612903225806, 'f1': 0.8938975840472448}
    def __init__(self, MODEL_NAME,num_classes):
        super().__init__()
        self.history_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.response_encoder = AutoModel.from_pretrained(MODEL_NAME)
        

        self.classifier = nn.Sequential(
            nn.Linear(768*5, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, history_ids, history_mask, response_ids, response_mask):
        # 历史编码
        hist_outputs = self.history_encoder(
            input_ids=history_ids,
            attention_mask=history_mask
        )
        hist_embed = hist_outputs.last_hidden_state[:, 0, :]
        
        # 回复编码
        resp_outputs = self.response_encoder(
            input_ids=response_ids,
            attention_mask=response_mask
        )
        resp_embed = resp_outputs.last_hidden_state[:, 0, :]
        
        # 特征融合
        combined = torch.cat([hist_embed, resp_embed,hist_embed+resp_embed,hist_embed*resp_embed,torch.abs(hist_embed-resp_embed)], dim=1)
        return self.classifier(combined)


class DualEncoder0(nn.Module): # {'accuracy': 0.8951612903225806, 'f1': 0.8938975840472448}
    def __init__(self, MODEL_NAME,num_classes):
        super().__init__()
        self.history_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.response_encoder = AutoModel.from_pretrained(MODEL_NAME)
        
        # 冻结前8层参数
        # for layer in self.history_encoder.encoder.layer[:8]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

        # for layer in self.response_encoder.encoder.layer[:8]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
                
        self.classifier = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, history_ids, history_mask, response_ids, response_mask):
        # 历史编码
        hist_outputs = self.history_encoder(
            input_ids=history_ids,
            attention_mask=history_mask
        )
        hist_embed = hist_outputs.last_hidden_state.mean(dim=1).squeeze(1)
        
        # 回复编码
        resp_outputs = self.response_encoder(
            input_ids=response_ids,
            attention_mask=response_mask
        )
        resp_embed = resp_outputs.last_hidden_state.mean(dim=1).squeeze(1)
        # 特征融合
        combined = torch.cat([hist_embed, resp_embed], dim=1)
        return self.classifier(combined)

class DualEncoder1(nn.Module):
    def __init__(self, MODEL_NAME, num_classes):
        super().__init__()
        self.history_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.response_encoder = AutoModel.from_pretrained(MODEL_NAME)

        # 冻结策略调整：仅冻结前6层
        for encoder in [self.history_encoder, self.response_encoder]:
            for layer in encoder.encoder.layer[:6]:
                for param in layer.parameters():
                    param.requires_grad = False

        # 共享投影层参数
        self.projection = nn.Sequential(
            nn.Linear(768*2, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Tanh()
        )

        # 增强型分类器
        self.classifier = nn.Sequential(
            nn.Linear(256*4, 512),  # 包含交互特征
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, history_ids, history_mask, response_ids, response_mask):
       
        def enhanced_pooling(outputs):
            cls_embed = outputs.last_hidden_state[:, 0, :]
            mean_embed = outputs.last_hidden_state.mean(dim=1)
    
            return torch.cat([cls_embed, mean_embed], dim=1)
        
        hist_outputs = self.history_encoder(history_ids, history_mask)
        
        hist_embed = self.projection(enhanced_pooling(hist_outputs))
        
        resp_outputs = self.response_encoder(response_ids, response_mask)
        resp_embed = self.projection(enhanced_pooling(resp_outputs))

       
        combined = torch.cat([
            hist_embed,
            resp_embed,
            hist_embed - resp_embed,
            hist_embed * resp_embed
        ], dim=1)

        return self.classifier(combined)

class ResidualBlock(nn.Module):
    
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.GELU(),
            nn.LayerNorm(size),
            nn.Dropout(0.1))
    
    def forward(self, x):
        return x + self.block(x)


class TutorModel(torch.nn.Module):
    def __init__(self, num_tutors, num_experts=4, expert_capacity=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained('deberta-v3-base')
        
        # MoE层参数
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # 共享的MoE层
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(768, expert_capacity),
                torch.nn.GELU(),
                torch.nn.LayerNorm(expert_capacity),
                torch.nn.Linear(expert_capacity, 768)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = torch.nn.Linear(768, num_experts)
        
        # 任务特定头部
        self.mistake_id = torch.nn.Linear(768, 3)
        self.mistake_loc = torch.nn.Linear(768, 3)
        self.guidance = torch.nn.Linear(768, 3)
        self.action = torch.nn.Linear(768, 3)
        self.tutor_id = torch.nn.Linear(768, num_tutors)
        
        # 共享的dropout和归一化
        self.dropout = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS] token
        
        # MoE处理
        gate_scores = torch.softmax(self.gate(pooled_output), dim=-1)
        expert_outputs = []
        
        for expert in self.experts:
            expert_outputs.append(expert(pooled_output))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden_size]
        moe_output = torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=1)
        
        # 共享处理
        moe_output = self.layer_norm(moe_output)
        moe_output = self.dropout(moe_output)
        
        # 任务特定输出
        return {
            'mistake_id': self.mistake_id(moe_output),
            'mistake_loc': self.mistake_loc(moe_output),
            'guidance': self.guidance(moe_output),
            'action': self.action(moe_output),
            'tutor_id': self.tutor_id(moe_output)
        }

class Task3Encoder(nn.Module):
    def __init__(self, MODEL_NAME,num_classes=3):
        super().__init__()
        self.history_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.response_encoder = AutoModel.from_pretrained(MODEL_NAME)
        
        # 冻结前8层参数
        # for layer in self.history_encoder.encoder.layer[:8]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

        # for layer in self.response_encoder.encoder.layer[:8]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
                
        self.classifier = nn.Sequential(
            
            nn.Linear(1024*2, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
        
    def forward(self, history_ids, history_mask, response_ids, response_mask):
        # 历史编码
        hist_outputs = self.history_encoder(
            input_ids=history_ids,
            attention_mask=history_mask
        )
        hist_embed = hist_outputs.last_hidden_state[:, 0, :]
       
        
        # 回复编码
        resp_outputs = self.response_encoder(
            input_ids=response_ids,
            attention_mask=response_mask
        )
        resp_embed = resp_outputs.last_hidden_state[:, 0, :]
        
        # 特征融合
        combined = torch.cat([hist_embed, resp_embed], dim=1)
        return self.classifier(combined)



class Task3Encoder1(nn.Module):
    def __init__(self, MODEL_NAME,num_classes=3):
        super().__init__()
        self.history_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.response_encoder = AutoModel.from_pretrained(MODEL_NAME)
        
        # 冻结前8层参数
        # for layer in self.history_encoder.encoder.layer[:8]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

        # for layer in self.response_encoder.encoder.layer[:8]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
                
        self.classifier = nn.Sequential(
            
            nn.Linear(768*2, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, history_ids, history_mask, response_ids, response_mask):
        # 历史编码
        hist_outputs = self.history_encoder(
            input_ids=history_ids,
            attention_mask=history_mask
        )
        hist_embed = hist_outputs.last_hidden_state
        hist_embed=  hist_embed.mean(dim=1).squeeze(1)

        
        # 回复编码
        resp_outputs = self.response_encoder(
            input_ids=response_ids,
            attention_mask=response_mask
        )
        resp_embed = resp_outputs.last_hidden_state
        resp_embed =  resp_embed.mean(dim=1).squeeze(1)
        
        # 特征融合
        combined = torch.cat([hist_embed, resp_embed], dim=1)
        return self.classifier(combined)


class Task3Encoder2(nn.Module):
    def __init__(self, MODEL_NAME, num_classes):
        super().__init__()
        self.history_response_encoder = AutoModel.from_pretrained(MODEL_NAME)
       
        
        self.projection = nn.Sequential(
            nn.Linear(768*2, 512),
            ResidualBlock(512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            ResidualBlock(256),
            nn.Tanh()
        )


        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 包含交互特征
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, history_ids, history_mask):
       
        def enhanced_pooling(outputs):
            cls_embed = outputs.last_hidden_state[:, 0, :]
            mean_embed = outputs.last_hidden_state.mean(dim=1)
    
            return torch.cat([cls_embed, mean_embed], dim=1)
        
        hist_outputs = self.history_response_encoder(history_ids, history_mask)
        
        hist_embed = self.projection(enhanced_pooling(hist_outputs))
        
    

        return self.classifier(hist_embed)