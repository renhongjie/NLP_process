import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.bert=args.bert_model
        #True是训练，False是固化
        for param in self.bert.parameters():
            param.requires_grad=True
        #bert后面接上游任务，其实就是把最后接线性层，把类别化为自己的类别
        self.fc=nn.Linear(args.hidden_size,args.num_classes)
    def forward(self,x,mask):
        #x [ids,seq_len,mask]
        context=x#对应输入的句子,shape=[128/批次,32/句子长度]
        mask=mask#对padding部分进行mask,shape=[128/批次,32/句子长度]
        self.bert(context)
        _,pooled=self.bert(context,attention_mask=mask) #shape=[128/批次,768/隐藏层个数]
        out=self.fc(pooled)#shape=[128/批次 ,10/类别个数]
        return out