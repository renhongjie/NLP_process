import torch
import torch.nn as nn
import torch.nn.functional as F
class textCNN(nn.Module):
    def __init__(self, args,weight):
        super(textCNN, self).__init__()
        self.embedding_S = nn.Embedding(args.vocab_size+1, args.embed_size)
        self.embedding_S.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding_S.weight.requires_grad = False
        self.embedding_D = nn.Embedding(args.vocab_size+1, args.embed_size)
        self.embedding_D.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding_D.weight.requires_grad = True
        num_filters = 256
        self.classification_num = args.classification_num
        self.conv1 = nn.Conv2d(1,num_filters,(2,args.embed_size))
        self.conv2 = nn.Conv2d(1,num_filters,(3,args.embed_size))
        self.conv3 = nn.Conv2d(1,num_filters,(4,args.embed_size))
        self.conv4 = nn.Conv2d(1,num_filters,(5,args.embed_size))
        self.pool1 = nn.MaxPool2d((args.seq_len - 2 + 1, 1))
        self.pool2 = nn.MaxPool2d((args.seq_len - 3 + 1, 1))
        self.pool3 = nn.MaxPool2d((args.seq_len - 4 + 1, 1))
        self.pool4 = nn.MaxPool2d((args.seq_len - 5 + 1, 1))
        self.dropout = nn.Dropout(args.droput)
        self.linear = nn.Linear(2*num_filters*4,args.classification_num)
    def forward(self, x):
        #两层通道
        out1 = self.embedding_S(x).view(x.shape[0],1,x.shape[1],-1)
        out2 = self.embedding_D(x).view(x.shape[0],1,x.shape[1],-1)
        #对第一层进行卷积
        x1 = F.relu(self.conv1(out1))
        x2 = F.relu(self.conv2(out1))
        x3 = F.relu(self.conv3(out1))
        x4 = F.relu(self.conv4(out1))
        #对第一层进行池化
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)
        #对第二层进行卷积
        x5 = F.relu(self.conv1(out2))
        x6 = F.relu(self.conv2(out2))
        x7 = F.relu(self.conv3(out2))
        x8 = F.relu(self.conv4(out2))
        #对第二层进行池化
        x5 = self.pool1(x5)
        x6 = self.pool2(x6)
        x7 = self.pool3(x7)
        x8 = self.pool4(x8)
        #合体～
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),1)
        x = x.view(x.shape[0], 1, -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = x.view(-1, self.classification_num)
        return x
