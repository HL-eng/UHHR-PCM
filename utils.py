import torch
import math
import numpy as np
import os
import pickle
from torch import bmm

from torch.autograd import Variable


def cal_similarity(object_node_feat, neigh_node_feat, neigh_nodes, layers):
    '''
    :param object_node_feat: 目标结点的特征
    :param neigh_node_feat:  目标结点的所有邻居结点特征
    :param probability:      取样概率 其实这个参数可以由下面的layers得到 p = (1/2)的j-1次方
    :param layers:           图神经网络的第n层，
    :param neigh_nodes:      目标结点的邻居数量，
    :return:                 返回的是一个前k个相似度索引，以方便进行下一次的聚合操作
    '''
    # object_node_feat.cuda()
    # neigh_node_feat.cuda()
    similarity = []  # 用于保存目标结点与邻居结点的相似度
    p = pow(1 / 2, layers)  # 取样概率
    sample_nums = math.ceil(neigh_nodes * p)  # 取样数量  不加取样的那一行直接换成0就行  neigh_nodes就是邻居数量，也就是目标结点的度
    for i in range(neigh_nodes):
        sim = torch.mul(object_node_feat, neigh_node_feat[i]).sum(dim=0)
        sim = sim.detach().cpu()
        similarity.append(sim)
    # 返回前n个大的值
    index = np.argsort(similarity)[-sample_nums:].tolist()
    return index, sample_nums


def load_visual_text(path_visual):
   with open(path_visual,'rb') as fr:
      visual_feat = pickle.load(fr)
   return visual_feat


# 遍历文件夹下的文件,直接返回其列表就行，到时候直接引用 0：user_clo 1:user_user 2:clo_user 3:clo_clo
def traversal(path):
    Filelist = []
    graph_list = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    # 然后再对这个文件列表进行遍历，读取出全部的交互数据图
    print(Filelist)
    for i in Filelist:
        with open(i, 'rb') as fr:
            graph_list.append(pickle.load(fr))
    return graph_list


def overfit(Us, Is, Js, Ks, model):
    return model.fit(set(Us), set(Is + Js + Ks))


def bpr(data, model, v_feat, t_feat, mode='train', emb_dim=64, model2=None):
    Us = [int(pair[0]) for pair in data]  # 目标用户
    Is = [int(pair[1]) for pair in data]  # 上衣结点
    Js = [int(pair[2]) for pair in data]  # 下衣正例结点
    Ks = [int(pair[3]) for pair in data]  # 下衣负例结点

    f_u_v_from_uc, f_u_t_from_uc = model.forward_u(Us, idx=2, mode=mode, v_feat=v_feat, t_feat=t_feat)
    #f_u_v_from_uu, f_u_t_from_uu = model.forward_u(Us, idx=3, mode=mode)
    #f_u_v = 0.6 * f_u_v_from_uc + 0.4 * f_u_v_from_uu
    #f_u_t = 0.6 * f_u_t_from_uc + 0.4 * f_u_t_from_uu

    f_u_v = f_u_v_from_uc
    f_u_t = f_u_t_from_uc

    f_I_v_feat_cc, f_I_t_feat_cc, f_J_v_feat_cc, \
    f_J_t_feat_cc, f_K_v_feat_cc, f_K_t_feat_cc = model.forward_c(Is, Js, Ks, idx=0, mode=mode, v_feat=v_feat, t_feat=t_feat)

    f_I_v_feat_cu, f_I_t_feat_cu, f_J_v_feat_cu, \
    f_J_t_feat_cu, f_K_v_feat_cu, f_K_t_feat_cu = model.forward_c(Is, Js, Ks, idx=1, mode=mode, v_feat=v_feat, t_feat=t_feat)

    f_top_v = 0.8 * f_I_v_feat_cu + 0.2 * f_I_v_feat_cc
    f_top_t = 0.5 * f_I_t_feat_cu + 0.5 * f_I_t_feat_cc

    f_pob_v = 0.8 * f_J_v_feat_cu + 0.2 * f_J_v_feat_cc
    f_pob_t = 0.5 * f_J_t_feat_cu + 0.5 * f_J_t_feat_cc

    f_pab_v = 0.8 * f_K_v_feat_cu + 0.2 * f_K_v_feat_cc
    f_pab_t = 0.5 * f_K_t_feat_cu + 0.5 * f_K_t_feat_cc

    visual_ij = bmm(f_top_v.unsqueeze(1), f_pob_v.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    text_ij = bmm(f_top_t.unsqueeze(1), f_pob_t.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    visual_ik = bmm(f_top_v.unsqueeze(1), f_pab_v.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    text_ik = bmm(f_top_t.unsqueeze(1), f_pab_t.unsqueeze(-1)).squeeze(-1).squeeze(-1)


    p_ij = 0.5 * visual_ij + 0.5 * text_ij
    p_ik = 0.5 * visual_ik + 0.5 * text_ik


    batchsize = len(f_u_v)  # 注意要把下面的512改成向量拼接之后的维度
    cuj = bmm(f_u_v.view(batchsize, 1, emb_dim), f_pob_v.view(batchsize, emb_dim, 1)).view(batchsize) + \
          bmm(f_u_t.view(batchsize, 1, emb_dim), f_pob_t.view(batchsize, emb_dim, 1)).view(batchsize)
    cuk = bmm(f_u_v.view(batchsize, 1, emb_dim), f_pab_v.view(batchsize, emb_dim, 1)).view(batchsize) + \
          bmm(f_u_t.view(batchsize, 1, emb_dim), f_pab_t.view(batchsize, emb_dim, 1)).view(batchsize)

    # 增加一个判断 如果是train方法，就增加返回正则项那一项，如果是验证测试方法，就不返回正则化的那一项
    if mode == 'train':
        # 定义一个方法  讲，各个服装I，J，K以及用户传入到此
        cukjweight = overfit(Us, Is, Js, Ks, model2)
        return 0.3 * p_ij + 0.7 * cuj - 0.3 * p_ik - 0.7 * cuk, cukjweight
    if mode == 'valid':
        return 0.3 * p_ij + 0.7 * cuj - 0.3 * p_ik - 0.7 * cuk