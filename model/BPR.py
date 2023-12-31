import torch.nn as nn
from torch.nn import *
import torch

class BPR(nn.Module):
    def __init__(self, user_set, item_set, hidden_dim=512):
        super(BPR, self).__init__()
        self.hidden_dim = hidden_dim
        # 下面这些矩阵都是随机初始化的
        self.user_gama = Embedding(len(user_set), self.hidden_dim)
        self.item_gama = Embedding(len(item_set), self.hidden_dim)
        self.user_beta = Embedding(len(user_set), 1)
        self.item_beta = Embedding(len(item_set), 1)
        self.theta_user_visual = Embedding(len(user_set), self.hidden_dim)
        self.theta_user_text = Embedding(len(user_set), self.hidden_dim)

        self.user_set = list(user_set)
        self.item_set = list(item_set)

        init.uniform_(self.theta_user_text.weight, 0, 0.01)  # 相当于对权重从均匀分布当中进行采样
        init.uniform_(self.theta_user_visual.weight, 0, 0.01)  # 相当于对偏置从均匀分布当中进行采样
        init.uniform_(self.user_gama.weight, 0, 0.01)
        init.uniform_(self.user_beta.weight, 0, 0.01)
        init.uniform_(self.item_gama.weight, 0, 0.01)
        init.uniform_(self.item_beta.weight, 0, 0.01)
        # 字典类型 key:userId/ItemId value:下标  注意user_set以及item_set的内容
        self.user_idx = {user: _index for _index, user in enumerate(user_set)}
        self.item_idx = {item: _index for _index, item in enumerate(item_set)}

    def get_user_idx(self, users):
        if self.user_beta.weight.is_cuda:
            return torch.tensor([self.user_idx[user] for user in users]) \
                .long() \
                .cuda(self.user_beta.weight.get_device())
        else:
            return torch.tensor([self.user_idx[user] for user in users]) \
                .long()
    def get_item_idx(self, items):
        if self.user_beta.weight.is_cuda:
            return torch.tensor([self.item_idx[str(item)] for item in items]) \
                .long() \
                .cuda(self.user_beta.weight.get_device())
        else:
            return torch.tensor([self.item_idx[str(item)] for item in items]) \
                .long()
    def get_user_gama(self, users):
        return self.user_gama(self.get_user_idx(users))

    def get_item_gama(self, items):
        return self.item_gama(self.get_item_idx(items))

    def get_theta_user_visual(self, users):
        return self.theta_user_visual(self.get_user_idx(users)) #继承自BPR当中的方法get_user_idx

    def get_theta_user_text(self, users):
        return self.theta_user_text(self.get_user_idx(users))

    def fit(self, users, items, p=2):
        user_gama = self.get_user_gama(users)
        user_beta = self.user_beta(self.get_user_idx(users))
        item_gama = self.get_item_gama(items)
        item_beta = self.item_beta(self.get_item_idx(items))
        user_text = self.get_theta_user_text(set(users))
        user_visual = self.get_theta_user_visual(set(users))
        return user_gama.norm(p=p)+ item_beta.norm(p=p)+ user_beta.norm(p=p)+item_gama.norm(p=p) \
               + user_text.norm(p=p) + user_visual.norm(p=p)