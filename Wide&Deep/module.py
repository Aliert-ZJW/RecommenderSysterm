import torch.nn as nn
import torch
import torch.nn.functional as F


class Linear(nn.Module):
    """
    Linear part
    """

    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


class Dnn(nn.Module):
    """
    Dnn part
    """

    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout: 失活率
        """
        super(Dnn, self).__init__()

        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)

        x = self.dropout(x)
        return x


class WideDeep(nn.Module):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0.):
        super(WideDeep, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0, len(self.dense_feature_cols) + len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
                                'embed_dim'])
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols))
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_input, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)

        # Wide,在这里因为没有那么多数据进行区分，所以在wide部分使用数值型数据代替
        wide_out = self.linear(dense_input)

        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)

        # out
        outputs = F.sigmoid(0.5 * (wide_out + deep_out))

        return outputs