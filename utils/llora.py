import torch
import torch.nn as nn
import math

class LoRAMultiheadAttention(nn.Module):
    def __init__(self, attn, rank = 8, alpha = 16.0):
        super().__init__()
        self.attn    = attn
        self.d       = attn.embed_dim
        self.scaling = alpha / rank
        
        for p in self.attn.parameters():
            p.requires_grad = False
        
        self.lora_q_A = nn.Parameter(torch.empty(rank, self.d))
        self.lora_q_B = nn.Parameter(torch.zeros(self.d, rank))
        self.lora_k_A = nn.Parameter(torch.empty(rank, self.d))
        self.lora_k_B = nn.Parameter(torch.zeros(self.d, rank))
        self.lora_v_A = nn.Parameter(torch.empty(rank, self.d))
        self.lora_v_B = nn.Parameter(torch.zeros(self.d, rank))
        
        for p in [self.lora_q_A, self.lora_k_A, self.lora_v_A]:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    
    def _delta(self, x, A, B):
        return (x @ A.T @ B.T) * self.scaling
    
    def forward(self, query, key, value, **kwargs):
        query = query + self._delta(query, self.lora_q_A, self.lora_q_B)
        key   = key   + self._delta(key,   self.lora_k_A, self.lora_k_B)
        value = value + self._delta(value, self.lora_v_A, self.lora_v_B)
        return self.attn(query, key, value, **kwargs)


def apply_lora_attention(model, rank=8, alpha=16.0, patch_emb=False, positional_encoding=False):
    
    for p in model.parameters():
        p.requires_grad = False
    
    for i, layer in enumerate(model.backbone.encoder.layers):
        layer.self_attn_s = LoRAMultiheadAttention(layer.self_attn_s, rank=rank, alpha=alpha)
        layer.self_attn_t = LoRAMultiheadAttention(layer.self_attn_t, rank=rank, alpha=alpha)
    
    for p in model.classifier.parameters():
        p.requires_grad = True

    if patch_emb : 
        for p in model.backbone.patch_embedding.parameters():
            p.requires_grad = True

    if positional_encoding : 
        for p in model.backbone.patch_embedding.positional_encoding.parameters():
            p.requires_grad = True
    
    model = model.to('cuda')
    
    return model
