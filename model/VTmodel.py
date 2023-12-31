import torch
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn import *
from torch.nn.init import uniform_
'''
TextCNN用于对文本特征进行处理
'''
class TextCNN(Module):
    def __init__(self, sentence_size = (83, 300), output_size = 128, uniform=False):
        super(TextCNN, self).__init__()
        self.max_sentense_length, self.word_vector_size = sentence_size
        self.text_cnn = ModuleList([Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(2,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 1,1),stride=1)
        ), Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(3,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 2,1),stride=1)
        ), Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(4,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 3,1),stride=1)
        ), Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(5,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 4,1),stride=1)
        )])
        self.text_nn = Sequential(
            Linear(400,output_size),
            Sigmoid(),
        )
        if uniform == True:
            for i in range(4):
                init.uniform_(self.text_cnn[i][0].weight.data, 0, 0.001).cuda()
                init.uniform_(self.text_cnn[i][0].bias.data, 0, 0.001).cuda()
            init.uniform_(self.text_nn[0].weight.data, 0, 0.001).cuda()
            init.uniform_(self.text_nn[0].bias.data, 0, 0.001).cuda()
    def forward(self, input):
        return self.text_nn(
            cat([conv2d(input).squeeze_(-1).squeeze_(-1) for conv2d in self.text_cnn], 1)
        )

class VisualProcess(Module):
    def __init__(self,embedding_weight ,  #此处的用户集合与衣服集合是集合当中的全部
        max_sentence = 83,  text_feature_dim=300,
        visual_feature_dim = 768, hidden_dim=128,
        uniform_value = 0.5):
        super(VisualProcess, self).__init__()
        self.uniform_value = uniform_value
        self.hidden_dim = hidden_dim
        self.visual_nn = Sequential(
            Linear(visual_feature_dim, 1024),  # 2048->512
            Sigmoid(),
            Linear(1024, self.hidden_dim),  # 2048->512
            Sigmoid(),
        )
        self.visual_nn[0].apply(lambda module: uniform_(module.weight.data, 0, 0.001)).cuda()
        self.visual_nn[0].apply(lambda module: uniform_(module.bias.data, 0, 0.001)).cuda()

        self.visual_nn[2].apply(lambda module: uniform_(module.weight.data, 0, 0.001)).cuda()
        self.visual_nn[2].apply(lambda module: uniform_(module.bias.data, 0, 0.001)).cuda()


        #print('generating user & item Parmeters')

        # load text features
        self.max_sentense_length = max_sentence  # 最大句子长度  83

        # text embedding layer   相当于是使用预训练好的词向量
        self.text_embedding = Embedding.from_pretrained(embedding_weight, freeze=False)

        '''
            text features embedding layers
        '''
        self.textcnn = TextCNN(sentence_size=(max_sentence, text_feature_dim), output_size=hidden_dim)
    def forward_u(self, neigh, visual_features, text_features):

        #对视觉特征以及文本特征进行处理
        if not self.visual_nn[0].weight.data.is_cuda:
            neigh_visual_latent = self.visual_nn(cat(                  #下衣负例视觉特征
                [visual_features[str(n)].unsqueeze(0) for n in neigh], 0
            ))

            neigh_text_latent = self.textcnn(                     #下衣负例的文本特征 使用了text_embedding
                self.text_embedding(
                    cat(
                        [text_features[str(n)].unsqueeze(0) for n in neigh], 0
                    )
                ) .unsqueeze_(1)
            )

        else :
            neigh_visual_latent = self.visual_nn(cat(  # 下衣负例视觉特征
                [visual_features[str(n)].unsqueeze(0) for n in neigh], 0
            ).cuda())

            neigh_text_latent = self.textcnn(  # 下衣负例的文本特征 使用了text_embedding
                self.text_embedding(
                    cat(
                        [text_features[str(n)].unsqueeze(0) for n in neigh], 0
                    ).cuda()
                ).unsqueeze_(1)
            )
        return neigh_visual_latent, neigh_text_latent

    #
    def forward_c(self, Is, Js, Ks, visual_features, text_features):
        if not self.visual_nn[0].weight.data.is_cuda:
            # 注意使用unsequeeze方法后，张量的变化形式
            I_visual_latent = self.visual_nn(cat(  # 上衣视觉特征
                [visual_features[str(I)].unsqueeze(0) for I in Is], 0
            ))
            J_visual_latent = self.visual_nn(cat(  # 下衣正例视觉特征
                [visual_features[str(J)].unsqueeze(0) for J in Js], 0
            ))
            K_visual_latent = self.visual_nn(cat(  # 下衣负例视觉特征
                [visual_features[str(K)].unsqueeze(0) for K in Ks], 0
            ))

            I_text_latent = self.textcnn(  # 上衣的文本特征 使用了text_embedding
                self.text_embedding(
                    cat(
                        [text_features[str(I)].unsqueeze(0) for I in Is], 0
                    )
                ).unsqueeze_(1)
            )
            J_text_latent = self.textcnn(  # 下衣正例的文本特征 使用了text_embedding
                self.text_embedding(
                    cat(
                        [text_features[str(J)].unsqueeze(0) for J in Js], 0
                    )
                ).unsqueeze_(1)
            )
            K_text_latent = self.textcnn(  # 下衣负例的文本特征 使用了text_embedding
                self.text_embedding(
                    cat(
                        [text_features[str(K)].unsqueeze(0) for K in Ks], 0
                    )
                ).unsqueeze_(1)
            )

        else:
            with torch.cuda.device(self.visual_nn[0].weight.data.get_device()):
                stream1 = torch.cuda.Stream()
                stream2 = torch.cuda.Stream()
                I_visual_latent = self.visual_nn(cat(
                    [visual_features[str(I)].unsqueeze(0) for I in Is], 0
                ).cuda())
                with torch.cuda.stream(stream1):
                    J_visual_latent = self.visual_nn(cat(
                        [visual_features[str(J)].unsqueeze(0) for J in Js], 0
                    ).cuda())
                with torch.cuda.stream(stream2):
                    K_visual_latent = self.visual_nn(cat(
                        [visual_features[str(K)].unsqueeze(0) for K in Ks], 0
                    ).cuda())
                I_text_latent = self.textcnn(
                    self.text_embedding(
                        cat(
                            [text_features[str(I)].unsqueeze(0) for I in Is], 0
                        ).cuda()
                    ).unsqueeze_(1)
                )
                with torch.cuda.stream(stream1):
                    J_text_latent = self.textcnn(
                        self.text_embedding(
                            cat(
                                [text_features[str(J)].unsqueeze(0) for J in Js], 0
                            ).cuda()
                        ).unsqueeze_(1)
                    )
                with torch.cuda.stream(stream2):
                    K_text_latent = self.textcnn(
                        self.text_embedding(
                            cat(
                                [text_features[str(K)].unsqueeze(0) for K in Ks], 0
                            ).cuda()
                        ).unsqueeze_(1)
                    )

        return I_visual_latent, I_text_latent, J_visual_latent, J_text_latent, K_visual_latent, K_text_latent