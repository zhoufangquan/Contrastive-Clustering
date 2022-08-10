import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, args, bert):
        super(Network, self).__init__()
        self.bert = bert
        self.feature_dim = args.feature_dim
        self.cluster_num = args.class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size,
                      self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size,
                      self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        output_i = self.bert(
            x_i['input_ids'],
            token_type_ids=x_i['token_type_ids'],
            attention_mask=x_i['attention_mask']
        )
        _, h_i = output_i[0], output_i[1]
        output_j = self.bert(
            x_j['input_ids'],
            token_type_ids=x_j['token_type_ids'],
            attention_mask=x_j['attention_mask']
        )
        _, h_j = output_j[0], output_j[1]

        # 将所有的特征 映射 到一个超球体的表面
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        output = self.bert(
            x['input_ids'],
            token_type_ids=x['token_type_ids'],
            attention_mask=x['attention_mask']
        )
        _, h = output[0], output[1]
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
