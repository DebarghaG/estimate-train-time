import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, mp, dim, precision):

        super(TransformerEncoder, self).__init__()
        if precision == 'fp16':
            self.encoder = nn.Sequential(
                self.get_layernorm([dim], precision),
                nn.Linear(dim, dim*3 // mp).half(),
                nn.Linear(dim//mp, dim, bias=False).half(),
                self.get_layernorm([dim], precision),
                nn.Linear(dim, dim*4//mp, bias=False).half(),
                nn.Linear(dim*4//mp, dim, bias=False).half()
            )
        else:
            self.encoder = nn.Sequential(
                self.get_layernorm([dim], precision),
                nn.Linear(dim, dim*3 // mp),
                nn.Linear(dim//mp, dim, bias=False),
                self.get_layernorm([dim], precision),
                nn.Linear(dim, dim*4//mp, bias=False),
                nn.Linear(dim*4//mp, dim, bias=False)
            )

    def get_layernorm(self, dim, precision):
        layernorm = nn.LayerNorm(dim)
        if precision == 'fp16':
            layernorm.weight.data = layernorm.weight.data.half()
            layernorm.bias.data = layernorm.weight.data.half()
        return layernorm
    
    def forward(self, x):
        n = 0
        for layer in self.encoder:
            if n == 2:
                one_third = x.size(-1) // 3
                x = x[..., :one_third]

            x = layer(x)
            n = n + 1
        return x

    
class FirstStage(nn.Module):
    def __init__(self, mp, dim, num_layers, precision):
        super(FirstStage, self).__init__()
        
        self.partition_vocab_size =  math.ceil(50257 / (128 * mp)) * 128
        
        if precision == 'fp16':
            self.embedding = nn.Embedding(num_embeddings=self.partition_vocab_size, embedding_dim=dim).to(dtype=torch.float16)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.partition_vocab_size, embedding_dim=dim).to(dtype=torch.float32)

        self.encoders = nn.ModuleList([TransformerEncoder(mp, dim, precision) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        return x


class MiddleStage(nn.Module):
    def __init__(self, mp, dim, num_layers, precision):
        super(MiddleStage, self).__init__()
        self.encoders = nn.ModuleList([TransformerEncoder(mp, dim, precision) for _ in range(num_layers)])
    
    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x


class LastStage(nn.Module):
    def __init__(self, mp, dim, num_layers, precision):
        super(LastStage, self).__init__()
        
        self.partition_vocab_size =  math.ceil(50257 / (128 * mp)) * 128

        self.encoders = nn.ModuleList([TransformerEncoder(mp, dim, precision) for _ in range(num_layers)])

        self.layernorm = self.get_layernorm([dim], precision)

        if precision == 'fp16':
            self.final_linear = nn.Linear(dim, self.partition_vocab_size//mp, bias=False).half()
        else:
            self.final_linear = nn.Linear(dim, self.partition_vocab_size//mp, bias=False)


    def get_layernorm(self, dim, precision):
        layernorm = nn.LayerNorm(dim)
        if precision == 'fp16':
            layernorm.weight.data = layernorm.weight.data.half()
            layernorm.bias.data = layernorm.weight.data.half()
        return layernorm


    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        x = self.layernorm(x)
        x = self.final_linear(x)
        return x
