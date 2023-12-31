import argparse
import os
import torch
import torch.nn as nn
from torch.nn import init
from model.graphModel import GraphSage
from torch.nn.functional import logsigmoid
from utils import traversal
from utils import bpr
from utils import load_visual_text
import time
from model.BPR import BPR


my_config = {
    "visual_features_dict": "./feat/all_visual_feat",
    "textural_idx_dict": "./feat/textfeatures",
    "textural_embedding_matrix": "./feat/smallnwjc2vec",
    "train_data": r"./dataset/train.csv",
    "valid_data": r"./dataset/valid.csv",
    "data": r"./dataset/data.csv",
}

def load_csv_data(train_data_path):
    result = []
    with open(train_data_path,'r',encoding='utf-8-sig') as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result

def load_embedding_weight(device):
    jap2vec = torch.load(my_config['textural_embedding_matrix'])
    embeding_weight = []
    for jap, vec in jap2vec.items():
        embeding_weight.append(vec.tolist())
    embeding_weight.append(torch.zeros(300))
    embedding_weight = torch.tensor(embeding_weight, device=device)
    return embedding_weight


#模型训练  因为此处的model就相当于是那个图卷积模型
def train(model,device, vidual_features,text_features, train_loader, optimizer,overfit):

    model.train()
    model.to(device)
    for iteration,aBatch in enumerate(train_loader):
        #t1 = time.time()
        output, cukjweight = bpr(aBatch[0], model, vidual_features, text_features,model2=overfit)
        loss = (-logsigmoid(output)).sum() + 0.001*cukjweight
        #print("正向传播的时间:{}".format(time.time()-t1))
        if iteration % 10 == 0:
            print(loss)
        #t = time.time()
        loss.backward()
        #print("反向传播的时间:{}".format(time.time()-t))
        optimizer.step()
        optimizer.zero_grad()

def evaluating(model, epoch,visual_features, text_features, test_csv):
    model.eval()
    testData = load_csv_data(test_csv)
    pos = 0
    batch_s = 100
    with torch.no_grad():
        for i in range(0, len(testData), batch_s):
            #print("{}_Starting".format(i))
            data = testData[i:i + batch_s] if i + batch_s <= len(testData) else testData[i:]
            # output的维度应该与上面的batch_s属于同一纬度，比如100维，然后每一维与0做比较，如果大于0就判断正确，但是如果小于0就判断错误
            output = bpr(data,model,mode='valid',v_feat=visual_features, t_feat=text_features)
            pos += float(torch.sum(output.ge(0)))
            #print("{}_Finished".format(i))
        print("evaling process: ", test_csv, epoch, pos / len(testData))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Research on modeling method of user individuation Compatibility based on GNN ')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=128, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train')
    parser.add_argument('--h_p', default=0.5 ,help='hyper parameters')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    embed_dim = args.embed_dim

    train_data = load_csv_data(my_config['train_data'])
    data = load_csv_data(my_config['data'])
    user_set = set([i[0] for i in data])  # set集合作用，可以不让元素有所重复
    item_set = set()
    for i in train_data:
        item_set.add(str(int(i[1])))  # 存储下衣的正例
        item_set.add(str(int(i[2])))  # 存储下衣的正例
        item_set.add(str(int(i[3])))  # 存储下衣的负例

    #种随机种子
    #torch.manual_seed(1000)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True
    u2e_visual = nn.Embedding(len(user_set), embed_dim).to(device)
    u2e_text = nn.Embedding(len(user_set), embed_dim).to(device)
    #init.uniform_(u2e_visual.weight, 0, 0.01)
    #init.uniform_(u2e_text.weight, 0, 0.01)

    #加载训练数据集图
    trainPath = './graph/train'
    validPath = './graph/valid'

    train_graph = traversal(trainPath)
    valid_graph = traversal(validPath)

    visual_features = load_visual_text(my_config['visual_features_dict'])
    # visual_features = torch.load(my_config['visual_features_dict'], map_location=lambda a, b: a.cuda())

    text_features = torch.load(my_config['textural_idx_dict'], map_location=lambda a, b: a.cuda())

    embedding_weight = load_embedding_weight(device)  # 存储了每个日语单词的词向量


    train_data = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.int))
    # train_loader = DataLoader(train_data, batch_size= batch_size,shuffle=True, drop_last=True)
    #在dataloader当中指定线程数目
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    user_idx = {user: _index for _index, user in enumerate(user_set)}
    graphSage = GraphSage(u2e_visual, u2e_text,  embedding_weight, train_graph,
                          valid_graph, embed_dim, user_idx, device)
    overfit = BPR(user_set, item_set).to(device)
    optimizer = torch.optim.Adam(graphSage.parameters(),lr=args.lr)

    # evaluating(graphSage,0, visual_features, text_features,my_config['valid_data'])
    for epoch in range(args.epochs):
        train(graphSage,device, visual_features, text_features, train_loader, optimizer,overfit)
        #torch.save(graphSage.state_dict(), 'net_params.pth')
        #evaluating(graphSage,epoch, visual_features, text_features,my_config['train_data'])
        evaluating(graphSage,epoch, visual_features, text_features,my_config['valid_data'])
        #evaluating(graphSage,epoch, visual_features, text_features,my_config['test_data'])

if __name__ == '__main__':
    main()