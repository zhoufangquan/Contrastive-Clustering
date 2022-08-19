import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, args, bert):
        super(Network, self).__init__()
        self.bert = bert
        self.hidden_dim = self.bert.config.hidden_size
        self.feature_dim = args.feature_dim
        self.cluster_num = args.class_num
        self.instance_projector = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.cluster_num),
            # nn.Softmax(dim=1)
        )

        nn.init.trunc_normal_(self.cluster_projector[2].weight, std=0.02)  # 正态分布式 截断。  不在指定区间的数据会被重新分配
        nn.init.trunc_normal_(self.cluster_projector[5].weight, std=0.02)

    def forward(self, x_i, x_j, return_ci=True):
        output_i = self.bert(
            x_i['input_ids'],
            token_type_ids=x_i['token_type_ids'],
            attention_mask=x_i['attention_mask']
        )
        # _, h_i = output_i[0], output_i[1]
        h_i = torch.sum(output_i[0]*x_i['attention_mask'].unsqueeze(-1), dim=1) / torch.sum(x_i['attention_mask'].unsqueeze(-1), dim=1)
        
        output_j = self.bert(
            x_j['input_ids'],
            token_type_ids=x_j['token_type_ids'],
            attention_mask=x_j['attention_mask']
        )
        # _, h_j = output_j[0], output_j[1]
        h_j = torch.sum(output_j[0]*x_j['attention_mask'].unsqueeze(-1), dim=1) / torch.sum(x_j['attention_mask'].unsqueeze(-1), dim=1)

        # 将所有的特征 映射 到一个超球体的表面
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_j = self.cluster_projector(h_j)

        if return_ci:
            c_i = self.cluster_projector(h_i)
            return z_i, z_j, c_i, c_j
        else:
            return z_i, z_j, c_j

    def forward_c(self, x):
        output = self.bert(
            x['input_ids'],
            token_type_ids=x['token_type_ids'],
            attention_mask=x['attention_mask']
        )
        _, h = output[0], output[1]
        c = self.cluster_projector(h)
        c = nn.Softmax(c, dim=1)
        # c = torch.argmax(c, dim=1)
        return c
    
    def forward_zc(self, x):
        output = self.bert(
            x['input_ids'],
            token_type_ids=x['token_type_ids'],
            attention_mask=x['attention_mask']
        )
        _, h = output[0], output[1]
        z = normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        c = nn.Softmax(c, dim=1)
        return z, c

