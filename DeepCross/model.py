import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')



class Res_block(nn.Module):
    def __init__(self, hidden_unit, dim_stack):
        super(Res_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.relu = nn.Relu()
        self.linear2 = nn.Linear(hidden_unit, dim_stack)

    def forward(self, x):
        x1 = x.clone()
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        out = x+x1
        return out


class DeepCrossing(nn.Module):
    def __int__(self, feature_info, hidden_unit, dropout=0, embedding_dim=10, out_dim=1):
        super(DeepCrossing, self).__init__()
        self.dense_f, self.sparse_f, self.sparse_f_map = feature_info
        self.embedding_layers = nn.ModuleDict({
            'embedding_'+str(key): nn.Embedding(num_embeddings=val, embedding_dim=embedding_dim)
            for key, val in self.sparse_f_map.items()
        })
        embedding_all_dim = sum([embedding_dim]*len(self.sparse_f))
        dim_stack = len(self.dense_f) + embedding_all_dim
        self.res_layers = nn.ModuleList({
            Res_block(unit, dim_stack) for unit in hidden_unit
        })
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim_stack, out_dim)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()
        sparse_embedding = [self.embedding_layers['embedding_'+key](sparse_inputs[:, i])
                            for key, i in zip(self.sparse_f_map.keys(), range(sparse_inputs.shape[1]))]
        sparse_embedding = torch.cat(sparse_embedding, aixs=-1)
        stack = torch.cat([sparse_embedding, dense_inputs], axis=-1)
        r = stack
        for res in self.res_layers:
            r = res(r)
        r = self.dropout(r)
        out = self.F.sigmoid(self.linear(r))
        return out








