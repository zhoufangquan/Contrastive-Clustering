import torch
import torch.nn as nn
import math


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class InstanceLossBoost(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, batch_size, temperature, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device


    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        pass

    def forward(self, z_i, z_j, pseudo_label):
        n = z_i.shape[0]
        invalid_index = pseudo_label == -1  # 没有为标签的数据的索引
        
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to( self.device )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(self.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()
        
        mask = mask.repeat(2, 2)
        mask_eye = mask_eye.repeat(2, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n*2).view(-1, 1).to(self.device),
            0,
        )
        logits_mask *= 1 - mask  # 负例对
        mask_eye = mask_eye * logits_mask  # 正例对

        z = torch.cat((z_i, z_j), dim=0) 
        sim = torch.matmul(z, z.t()) / self.temperature  # z @ z.t() / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)  # 获取每一行的最大值, 并保持2*n行1列
        sim = sim - sim_max.detach()  #  这样做是为了防止上溢，因为后面要进行指数运算

        exp_sim_neg = torch.exp(sim) * logits_mask  # 得到只有负例相似对的矩阵
        log_sim = sim - torch.log(exp_sim_neg.sum(1, keepdim=True))  #  log_softmax(), 分子上 正负例对 都有

        # compute mean of log-likelihood over positive
        instance_loss = -(mask_eye * log_sim).sum(1) / mask_eye.sum(1)  # 去分子为正例对的数据 
        instance_loss = instance_loss.view(2, n).m






class ClusterLossBoost(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()