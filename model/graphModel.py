import torch.nn as nn
import torch
from model.Attention import Attention
from model.VTmodel import VisualProcess
from utils import cal_similarity
import torch.nn.functional as F


class GraphSage(nn.Module):

    def __init__(self, u2e_visual, u2e_text, embed_weight, traingraph, validgraph,
                 emb_dim, user_map, device):
        super(GraphSage, self).__init__()
        self.u2e_visual = u2e_visual
        self.u2e_text = u2e_text
        self.user_idx = user_map
        self.traingraph = traingraph
        self.validgraph = validgraph
        self.emb_dim = emb_dim
        self.device = device
        # 需要有一个线性层
        self.linear1 = nn.Linear(2 * self.emb_dim, 64).to(device)  #
        self.vtmodel = VisualProcess(embed_weight).to(device)
        self.att = Attention(emb_dim).to(device)

    def get_user_idx(self, users):
        if self.u2e_visual.weight.is_cuda:
            return torch.tensor([self.user_idx[user] for user in users]) \
                .long() \
                .cuda(self.u2e_visual.weight.get_device())
        else:
            return torch.tensor([self.user_idx[user] for user in users]) \
                .long()

    def get_user_visual(self, users):
        return self.u2e_visual(self.get_user_idx(users))  # 继承自BPR当中的方法get_user_idx

    def get_user_text(self, users):
        return self.u2e_text(self.get_user_idx(users))

    # 查询目标结点的邻居向量
    def queryNeigh_u(self, nodes, idx, mode='train'):
        if mode == 'train':
            neigh_nodes = [self.traingraph[idx][int(node)] for node in nodes]
        if mode == 'valid':
            neigh_nodes = [self.validgraph[idx][int(node)] for node in nodes]
        return neigh_nodes

    def queryNeigh_c(self, I, J, K, idx, mode='train'):
        if mode == 'train':
            neigh_nodes_I = [self.traingraph[idx][int(node)] for node in I]
            neigh_nodes_J = [self.traingraph[idx][int(node)] for node in J]
            neigh_nodes_K = [self.traingraph[idx][int(node)] for node in K]
        if mode == 'valid':
            neigh_nodes_I = [self.validgraph[idx][int(node)] for node in I]
            neigh_nodes_J = [self.validgraph[idx][int(node)] for node in J]
            neigh_nodes_K = [self.validgraph[idx][int(node)] for node in K]
        return neigh_nodes_I, neigh_nodes_J, neigh_nodes_K

    # 参与聚合信息
    def aggreInfo_u_c(self, nodes, neigh_nodes, v_feat, t_feat):
        # 调用上面的方法先找到邻居之后在进行聚合信息
        v_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        t_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        u_v_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        u_t_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = neigh_nodes[i]
            num_neighs = len(tmp_adj)
            # 获取邻居结点的视觉特征以及文本特征
            neigh_v_feat, neigh_t_feat = self.vtmodel.forward_u(tmp_adj, v_feat, t_feat)
            u_idx = self.user_idx[nodes[i]]  # nodes[i]是取出的一个张量，不能直接作为字典的键值
            u_visual_rep = self.u2e_visual.weight[u_idx]  # 通过用户映射表将顶点映射成下标用于取用户的特征向量
            u_v_embed_matrix[i] = u_visual_rep
            idx, samples_nums = cal_similarity(u_visual_rep, neigh_v_feat, num_neighs, 1)  # 返回取样结点的下标
            neigh_v_feat = neigh_v_feat[idx]
            att_w = self.att(neigh_v_feat, u_visual_rep, samples_nums)
            neigh_v_feat = torch.mm(neigh_v_feat.t(), att_w).t()
            # 对获得的邻居张量求均值
            # neigh_v_feat = neigh_v_feat.mean(dim=0)
            v_embed_matrix[i] = neigh_v_feat
            u_text_rep = self.u2e_text.weight[u_idx]
            u_t_embed_matrix[i] = u_text_rep
            idx, samples_nums = cal_similarity(u_text_rep, neigh_t_feat, num_neighs, 1)  # 返回取样结点的下标
            neigh_t_feat = neigh_t_feat[idx]
            att_w = self.att(neigh_t_feat, u_text_rep, samples_nums)
            att_history = torch.mm(neigh_t_feat.t(), att_w).t()  # 将注意力分数分别于两个向量相乘
            # neigh_t_feat = neigh_t_feat.mean(dim=0)
            # t_embed_matrix[i] = neigh_t_feat
            t_embed_matrix[i] = att_history

        return u_v_embed_matrix, u_t_embed_matrix, v_embed_matrix, t_embed_matrix

    def aggreInfo_u_u(self, nodes, neigh_nodes):
        # 调用上面的方法先找到邻居之后在进行聚合信息
        v_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        t_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        u_v_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        u_t_embed_matrix = torch.empty(len(nodes), self.emb_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = neigh_nodes[i]
            if len(tmp_adj) != 0:
                num_neighs = len(tmp_adj)

                e_u_visual = self.get_user_visual(tmp_adj)
                e_u_text = self.get_user_text(tmp_adj)

                # u_idx = self.user_map[nodes[i]]  #取下标
                u_visual_rep = self.get_user_visual([nodes[i]])  # 目标结点视觉特征
                u_text_rep = self.get_user_text([nodes[i]])  # 目标结点文本特征
                u_v_embed_matrix[i] = u_visual_rep
                u_t_embed_matrix[i] = u_text_rep
                idx, samples_nums = cal_similarity(u_visual_rep.squeeze(0), e_u_visual, num_neighs, 1)  # 返回取样结点的下标
                e_u_visual = e_u_visual[idx]
                att_w = self.att(e_u_visual, u_visual_rep, samples_nums)
                neigh_v_feat = torch.mm(e_u_visual.t(), att_w).t()
                # neigh_v_feat = e_u_visual.mean(dim=0)
                v_embed_matrix[i] = neigh_v_feat  # 将邻居聚集的特征保留在embed_matrix矩阵当中

                idx, samples_nums = cal_similarity(u_text_rep.squeeze(0), e_u_text, num_neighs, 1)  # 返回取样结点的下标
                e_u_text = e_u_text[idx]
                att_w = self.att(e_u_text, u_text_rep, samples_nums)
                neigh_t_feat = torch.mm(e_u_text.t(), att_w).t()  # 将注意力分数分别于两个向量相乘
                # neigh_t_feat = e_u_text.mean(dim=0)
                t_embed_matrix[i] = neigh_t_feat  # 将邻居聚集的特征保留在embed_matrix矩阵当中

            else:
                u_visual_rep = self.get_user_visual([nodes[i]])  # 目标结点视觉特征
                u_text_rep = self.get_user_text([nodes[i]])  # 目标结点文本特征
                v_embed_matrix[i] = u_visual_rep
                t_embed_matrix[i] = u_text_rep
                u_v_embed_matrix[i] = u_visual_rep
                u_t_embed_matrix[i] = u_text_rep
        return u_v_embed_matrix, u_t_embed_matrix, v_embed_matrix, t_embed_matrix

    # 参与聚合信息
    def aggreInfo_c_c(self, I, J, K, neigh_I, neigh_J, neigh_K, v_feat, t_feat):
        I_v_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_v_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_v_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        I_t_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_t_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_t_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        I_v_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_v_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_v_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        I_t_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_t_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_t_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        for i in range(len(I)):
            Is, Js, Ks = list(neigh_I[i]), list(neigh_J[i]), list(neigh_K[i])
            I_neigh_v, I_neigh_t, J_neigh_v, \
            J_neigh_t, K_neigh_v, K_neigh_t = self.vtmodel.forward_c(Is, Js, Ks, v_feat, t_feat)
            I_v, I_t, J_v, J_t, K_v, K_t = self.vtmodel.forward_c([I[i]], [J[i]], [K[i]], v_feat, t_feat)
            I_v_feat[i], I_t_feat[i], J_v_feat[i], J_t_feat[i], K_v_feat[i], K_t_feat[i] = I_v, I_t, J_v, J_t, K_v, K_t

            I_idx, I_samples_nums = cal_similarity(I_v.squeeze(0), I_neigh_v, len(Is), 1)
            J_idx, J_samples_nums = cal_similarity(J_v.squeeze(0), J_neigh_v, len(Js), 1)
            K_idx, K_samples_nums = cal_similarity(K_v.squeeze(0), K_neigh_v, len(Ks), 1)

            I_neigh_v, J_neigh_v, K_neigh_v = I_neigh_v[I_idx], J_neigh_v[J_idx], K_neigh_v[K_idx]
            att_w_I = self.att(I_neigh_v, I_v, I_samples_nums)
            I_neigh_v = torch.mm(I_neigh_v.t(), att_w_I).t()

            att_w_J = self.att(J_neigh_v, J_v, J_samples_nums)
            J_neigh_v = torch.mm(J_neigh_v.t(), att_w_J).t()

            att_w_K = self.att(K_neigh_v, K_v, K_samples_nums)
            K_neigh_v = torch.mm(K_neigh_v.t(), att_w_K).t()

            # I_neigh_v, J_neigh_v, K_neigh_v = I_neigh_v.mean(dim=0), J_neigh_v.mean(dim=0), K_neigh_v.mean(dim=0)
            I_v_embed_matrix[i], J_v_embed_matrix[i], K_v_embed_matrix[i] \
                = I_neigh_v, J_neigh_v, K_neigh_v

            I_idx, I_samples_nums = cal_similarity(I_t.squeeze(0), I_neigh_t, len(Is), 1)
            J_idx, J_samples_nums = cal_similarity(J_t.squeeze(0), J_neigh_t, len(Js), 1)
            K_idx, K_samples_nums = cal_similarity(K_t.squeeze(0), K_neigh_t, len(Ks), 1)
            I_neigh_t, J_neigh_t, K_neigh_t = I_neigh_t[I_idx], J_neigh_t[J_idx], K_neigh_t[K_idx]

            att_w_I = self.att(I_neigh_t, I_t, I_samples_nums)
            I_neigh_t = torch.mm(I_neigh_t.t(), att_w_I).t()

            att_w_J = self.att(J_neigh_t, J_t, J_samples_nums)
            J_neigh_t = torch.mm(J_neigh_t.t(), att_w_J).t()

            att_w_K = self.att(K_neigh_t, K_t, K_samples_nums)
            K_neigh_t = torch.mm(K_neigh_t.t(), att_w_K).t()

            # I_neigh_t, J_neigh_t, K_neigh_t = I_neigh_t.mean(dim=0), J_neigh_t.mean(dim=0), K_neigh_t.mean(dim=0)
            I_t_embed_matrix[i], J_t_embed_matrix[i], K_t_embed_matrix[i] \
                = I_neigh_t, J_neigh_t, K_neigh_t

        return I_v_feat, I_t_feat, J_v_feat, J_t_feat, K_v_feat, K_t_feat, \
               I_v_embed_matrix, I_t_embed_matrix, J_v_embed_matrix, \
               J_t_embed_matrix, K_v_embed_matrix, K_t_embed_matrix

    def aggreInfo_c_u(self, I, J, K, neigh_I, neigh_J, neigh_K, v_feat, t_feat):
        I_v_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_v_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_v_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        I_t_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_t_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_t_embed_matrix = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        I_v_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_v_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_v_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        I_t_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        J_t_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        K_t_feat = torch.empty(len(I), self.emb_dim, dtype=torch.float).to(self.device)
        for i in range(len(I)):
            Is, Js, Ks = neigh_I[i], neigh_J[i], neigh_K[i]
            # 获取邻居结点的特征
            I_neighs_index = [self.user_idx[i] for i in Is]
            J_neighs_index = [self.user_idx[i] for i in Js]
            K_neighs_index = [self.user_idx[i] for i in Ks]
            I_neigh_v = self.u2e_visual.weight[I_neighs_index]
            J_neigh_v = self.u2e_visual.weight[J_neighs_index]
            K_neigh_v = self.u2e_visual.weight[K_neighs_index]
            I_neigh_t = self.u2e_text.weight[I_neighs_index]
            J_neigh_t = self.u2e_text.weight[J_neighs_index]
            K_neigh_t = self.u2e_text.weight[K_neighs_index]

            I_v, I_t, J_v, J_t, K_v, K_t = self.vtmodel.forward_c([I[i]], [J[i]], [K[i]], v_feat, t_feat)
            I_v_feat[i], I_t_feat[i], J_v_feat[i], J_t_feat[i], K_v_feat[i], K_t_feat[i] = I_v, I_t, J_v, J_t, K_v, K_t

            I_idx, I_samples_nums = cal_similarity(I_v.squeeze(0), I_neigh_v, len(Is), 1)
            J_idx, J_samples_nums = cal_similarity(J_v.squeeze(0), J_neigh_v, len(Js), 1)
            K_idx, K_samples_nums = cal_similarity(K_v.squeeze(0), K_neigh_v, len(Ks), 1)
            I_neigh_v, J_neigh_v, K_neigh_v = I_neigh_v[I_idx], J_neigh_v[J_idx], K_neigh_v[K_idx]

            att_w_I = self.att(I_neigh_v, I_v, I_samples_nums)
            I_neigh_v = torch.mm(I_neigh_v.t(), att_w_I).t()

            att_w_J = self.att(J_neigh_v, J_v, J_samples_nums)
            J_neigh_v = torch.mm(J_neigh_v.t(), att_w_J).t()

            att_w_K = self.att(K_neigh_v, K_v, K_samples_nums)
            K_neigh_v = torch.mm(K_neigh_v.t(), att_w_K).t()

            # I_neigh_v, J_neigh_v, K_neigh_v = I_neigh_v.mean(dim=0), J_neigh_v.mean(dim=0), K_neigh_v.mean(dim=0)
            I_v_embed_matrix[i], J_v_embed_matrix[i], K_v_embed_matrix[i] \
                = I_neigh_v, J_neigh_v, K_neigh_v

            I_idx, I_samples_nums = cal_similarity(I_t.squeeze(0), I_neigh_t, len(Is), 1)
            J_idx, J_samples_nums = cal_similarity(J_t.squeeze(0), J_neigh_t, len(Js), 1)
            K_idx, K_samples_nums = cal_similarity(K_t.squeeze(0), K_neigh_t, len(Ks), 1)
            I_neigh_t, J_neigh_t, K_neigh_t = I_neigh_t[I_idx], J_neigh_t[J_idx], K_neigh_t[K_idx]

            att_w_I = self.att(I_neigh_t, I_t, I_samples_nums)
            I_neigh_t = torch.mm(I_neigh_t.t(), att_w_I).t()

            att_w_J = self.att(J_neigh_t, J_t, J_samples_nums)
            J_neigh_t = torch.mm(J_neigh_t.t(), att_w_J).t()

            att_w_K = self.att(K_neigh_t, K_t, K_samples_nums)
            K_neigh_t = torch.mm(K_neigh_t.t(), att_w_K).t()

            # I_neigh_t, J_neigh_t, K_neigh_t = I_neigh_t.mean(dim=0), J_neigh_t.mean(dim=0), K_neigh_t.mean(dim=0)
            I_t_embed_matrix[i], J_t_embed_matrix[i], K_t_embed_matrix[i] \
                = I_neigh_t, J_neigh_t, K_neigh_t
        return I_v_feat, I_t_feat, J_v_feat, J_t_feat, K_v_feat, K_t_feat, \
               I_v_embed_matrix, J_v_embed_matrix, K_v_embed_matrix, \
               I_t_embed_matrix, J_t_embed_matrix, K_t_embed_matrix

    # 融合结点自身的特征进行聚合，减少图神经网络的过平滑性
    def add_self_loop_u(self, self_visual_rep, self_text_rep, v_neigh_feat, t_neigh_feat):
        # 不做张量拼接，而是做张量相加
        f_o_v_feat = torch.cat((self_visual_rep, v_neigh_feat), dim=1)
        # f_o_v_feat = 0.5 * self_visual_rep + 0.5 * v_neigh_feat
        # f_o_v_feat = F.leaky_relu(self.linear1(f_o_v_feat))
        f_o_v_feat = self.linear1(f_o_v_feat)
        f_o_v_feat = torch.sigmoid(f_o_v_feat)
        # f_o_v_feat = F.dropout(f_o_v_feat, p=0.5, training=self.training)

        f_o_t_feat = torch.cat((self_text_rep, t_neigh_feat), dim=1)
        # f_o_t_feat = 0.5 * self_text_rep + 0.5 * t_neigh_feat
        # f_o_t_feat = F.sigmoid(self.linear1(f_o_t_feat))
        f_o_t_feat = self.linear1(f_o_t_feat)
        f_o_t_feat = torch.sigmoid(f_o_t_feat)
        # f_o_t_feat = F.dropout(f_o_t_feat,p=0.5,training=self.training)
        return f_o_v_feat, f_o_t_feat

    def add_self_loop_c(self, I_v, I_t, J_v, J_t, K_v, K_t, I_v_neigh_feat, I_t_neigh_feat,
                        J_v_neigh_feat, J_t_neigh_feat,
                        K_v_neigh_feat, K_t_neigh_feat):
        # f_I_v_feat = 0.5 * I_v + 0.5 * I_v_neigh_feat
        # f_I_t_feat = 0.5 * I_t + 0.5 * I_t_neigh_feat
        # f_J_v_feat = 0.5 * J_v + 0.5 * J_v_neigh_feat
        # f_J_t_feat = 0.5 * J_t + 0.5 * J_t_neigh_feat
        # f_K_v_feat = 0.5 * K_v + 0.5 * K_v_neigh_feat
        # f_K_t_feat = 0.5 * K_t + 0.5 * K_t_neigh_feat

        f_I_v_feat = torch.cat((I_v, I_v_neigh_feat), dim=1)
        f_I_t_feat = torch.cat((I_t, I_t_neigh_feat), dim=1)
        f_J_v_feat = torch.cat((J_v, J_v_neigh_feat), dim=1)
        f_J_t_feat = torch.cat((J_t, J_t_neigh_feat), dim=1)
        f_K_v_feat = torch.cat((K_v, K_v_neigh_feat), dim=1)
        f_K_t_feat = torch.cat((K_t, K_t_neigh_feat), dim=1)

        f_I_v_feat = self.linear1(f_I_v_feat)
        f_I_v_feat = torch.sigmoid(f_I_v_feat)
        # f_I_v_feat = F.dropout(f_I_v_feat, p=0.5, training=self.training)
        f_I_t_feat = self.linear1(f_I_t_feat)
        f_I_t_feat = torch.sigmoid(f_I_t_feat)
        # f_I_t_feat = F.dropout(f_I_t_feat, p=0.5, training=self.training)
        f_J_v_feat = self.linear1(f_J_v_feat)
        f_J_v_feat = torch.sigmoid(f_J_v_feat)
        # f_J_v_feat = F.dropout(f_J_v_feat, p=0.5, training=self.training)
        f_J_t_feat = self.linear1(f_J_t_feat)
        f_J_t_feat = torch.sigmoid(f_J_t_feat)
        # f_J_t_feat = F.dropout(f_J_t_feat, p=0.5, training=self.training)
        f_K_v_feat = self.linear1(f_K_v_feat)
        f_K_v_feat = torch.sigmoid(f_K_v_feat)
        # f_K_v_feat = F.dropout(f_K_v_feat, p=0.5, training=self.training)
        f_K_t_feat = self.linear1(f_K_t_feat)
        f_K_t_feat = torch.sigmoid(f_K_t_feat)
        # f_K_t_feat = F.dropout(f_K_t_feat, p=0.5, training=self.training)
        return f_I_v_feat, f_I_t_feat, f_J_v_feat, f_J_t_feat, f_K_v_feat, f_K_t_feat

    def forward_u(self, batch_nodes_u, idx, mode, v_feat=None, t_feat=None):
        neigh_nodes = self.queryNeigh_u(batch_nodes_u, idx, mode)
        if idx == 2:  # user_clo
            self_v_rep, self_t_rep, v_neigh_features, t_neigh_features = self.aggreInfo_u_c(batch_nodes_u, neigh_nodes,
                                                                                            v_feat, t_feat)
        if idx == 3:  # user_user
            self_v_rep, self_t_rep, v_neigh_features, t_neigh_features = self.aggreInfo_u_u(batch_nodes_u, neigh_nodes)
        return self.add_self_loop_u(self_v_rep, self_t_rep, v_neigh_features, t_neigh_features)

    def forward_c(self, I, J, K, idx, mode, v_feat=None, t_feat=None):
        # 查找目标结点的邻居结点，之后把所有的邻居结点返回出来，进行信息的聚集
        neigh_I, neigh_J, neigh_K = self.queryNeigh_c(I, J, K, idx, mode)
        if idx == 0:  # clo_clo
            I_v, I_t, J_v, J_t, K_v, K_t, \
            I_v_neigh_feat, I_t_neigh_feat, J_v_neigh_feat, \
            J_t_neigh_feat, K_v_neigh_feat, K_t_neigh_feat = self.aggreInfo_c_c(I, J, K, neigh_I, neigh_J, neigh_K,
                                                                                v_feat, t_feat)
        if idx == 1:  # clo_user
            I_v, I_t, J_v, J_t, K_v, K_t, \
            I_v_neigh_feat, I_t_neigh_feat, J_v_neigh_feat, \
            J_t_neigh_feat, K_v_neigh_feat, K_t_neigh_feat = self.aggreInfo_c_u(I, J, K, neigh_I, neigh_J, neigh_K,
                                                                                v_feat, t_feat)

        return self.add_self_loop_c(I_v, I_t, J_v, J_t, K_v, K_t, I_v_neigh_feat, I_t_neigh_feat,
                                    J_v_neigh_feat, J_t_neigh_feat,
                                    K_v_neigh_feat, K_t_neigh_feat)
