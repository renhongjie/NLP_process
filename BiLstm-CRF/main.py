import argparse
import torch
import utils
import model
import train

#设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义超参数
parser = argparse.ArgumentParser()
#本地
#parser.add_argument('--data_path', type=str, default="/Users/ren/Desktop/nlp相关/实验1/aclImdb/")#文件路径
parser.add_argument('--data_path', type=str, default="/Users/ren/jupyter/NLP作业/实验2/data/")#文件路径
parser.add_argument('--embed_size', type=int, default=300)#embeding层宽度
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--num_hidens', type=int, default=100)
parser.add_argument('--seq_len', type=int, default=600)#文件长度，需要截断和填充
parser.add_argument('--batch_size', type=int, default=64)#批次
parser.add_argument('--bidirectional', type=bool, default=True)#是否开启双向
parser.add_argument('--classification_num', type=int, default=2)#分类个数
parser.add_argument('--lr', type=float, default=1e-4)#学习率
parser.add_argument('--dropout', type=float, default=0.5)#丢弃率
parser.add_argument('--num_epochs', type=int, default=100)#训练论数
parser.add_argument('--vocab_size', type=int, default=0)#vocab大小
parser.add_argument('--if_vail', type=bool, default=False)
parser.add_argument('--word2vec_path', type=str, default="/Users/ren/Desktop/nlp相关/nlp各种大文件/glove_to_word2vec.txt")#预训练词向量路径
parser.add_argument('--save_path', type=str, default="best.pth")#保存路径
parser.add_argument('--weight_decay', type=float, default=1e-4)#权重摔跤
args = parser.parse_args()

train_iter, test_iter, weight = utils.data(args)
print(weight)
net = model.NERLSTM_CRF(args, weight)
net=net.to(device)

train.train(args,device,net,train_iter,test_iter)

