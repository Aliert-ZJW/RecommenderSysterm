import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

# 首先构建一个两层的全连接网络，当然具体多少层可以自己控制
class DNN(nn.module):
    def __init__(self, hidden_dims, dropout):
        super(DNN, self).__init__()
        '''
        这里定义的hidden_units:列表， 每个元素表示每一层的神经单元个数，
        比如[256, 128, 64]，两层网络， 第一层神经单元128个，第二层64，注意第一个是输入维度
        hidden_dims[:-1] = (256, 128)
        hidden_dims[1:] = (128, 64) 大家可以根据自己的需求来自行定义
        '''
        self.linear_units = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in zip(hidden_dims[:-1], hidden_dims[1:])])
        self.relu = nn.relu()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        for linear in self.linear_units:
            x = linear(x)
            x = self.relu(x)
        x = self.dropout(x)
        return x

# 然后定义Product层
class Product(nn.Module):
    def __init__(self, mode, hidden_dims, embbedding_dim, feature_dim):
        super(Product, self).__init__()
        '''
        mode 是在P部分进行内积还是外积操作
        embbedding_dim 进行embedding之后的维度，也就是M
        feature_dim 原始特征有多少个域， 也就是N
        '''
        # z部分的初始化权重W
        self.w_z = nn.Parameter(torch.rand([feature_dim, embbedding_dim, hidden_dims[0]]), requires_grad=True)
        # 在P部分初始化权重根据mode的不同分成两部分
        if mode == 'in':
            self.w_p = nn.Parameter(torch.rand([feature_dim, feature_dim, hidden_dims[0]]), requires_grad=True)
        else:
            self.w_p = nn.Parameter(torch.rand([embbedding_dim, embbedding_dim, hidden_dims[0]]), requires_grad=True)

        self.l_b = torch.rand([hidden_dims[0]], requires_grad=True)

    def forward(self, z, sparse_embedding):
        l_z = torch.mm(z.reshape(z.shape[0], -1), self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)  # (None, hidden_dims[0])

        if self.mode == 'in':  # in模式  内积操作  p就是两两embedding先内积得到的[field_dim, field_dim]的矩阵
            p = torch.matmul(sparse_embedding, sparse_embedding.permute((0, 2, 1)))  # [None, field_num, field_num]
        else:  # 外积模式  这里的p矩阵是两两embedding先外积得到n*n个[embed_dim, embed_dim]的矩阵， 然后对应位置求和得到最终的1个[embed_dim, embed_dim]的矩阵
            # 所以这里实现的时候， 可以先把sparse_embeds矩阵在field_num方向上先求和， 然后再外积
            f_sum = torch.unsqueeze(torch.sum(sparse_embedding, dim=1), dim=1)  # [None, 1, embed_dim]
            p = torch.matmul(f_sum.permute((0, 2, 1)), f_sum)  # [None, embed_dim, embed_dim]

        l_p = torch.mm(p.reshape(p.shape[0], -1),
                       self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)  # [None, hidden_units[0]]

        output = l_p + l_z + self.l_b
        return output

class PNN(nn.module):

    def __init__(self, feature_info, hidden_dims, mode='in', dropout=0, embedding_dim=10, outdim=1):
        super(PNN, self).__init__()
        self.dense_feas, self.sparse_feas, self.sparse_feas_map = feature_info
        self.feature_num = len(self.sparse_feas)
        self.dense_num = len(self.dense_feas)
        self.mode = mode
        self.embed_dim = embedding_dim

        self.embedding_layers = nn.ModuleDict({
            'embedding_'+str(key): nn.Embedding(num_embedding=val, embedding_dim=embedding_dim)
            for key, val in self.sparse_feas_map.items()
        })

        self.product = Product(mode, hidden_dims, embedding_dim, self.feature_num)

        # 将数值型特征的维度加到DNN里面去
        hidden_dims[0] += self.dense_num
        self.dnn_network = DNN(hidden_dims, dropout)
        self.dnn_final = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]  # 数值型和类别型数据分开
        sparse_inputs = sparse_inputs.long()  # 需要转成长张量， 这个是embedding的输入要求格式
        sparse_embedding = [self.embedding_layers['embedding'+key](sparse_inputs[:, i])
                            for key, i in zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]
        sparse_embeds = torch.stack(sparse_embedding)
        sparse_embeds = sparse_embeds.permute(
            (1, 0, 2))  #  此时空间不连续， 下面改变形状不能用view，用reshape
        z = sparse_embeds

        # product layer
        sparse_inputs = self.product(z, sparse_embedding)

        # 把上面的连起来， 注意此时要加上数值特征
        l1 = nn.relu(torch.cat([sparse_inputs, dense_inputs], axis=-1))
        # dnn_network
        dnn_x = self.dnn_network(l1)

        outputs = nn.sigmoid(self.dense_final(dnn_x))

        return outputs



