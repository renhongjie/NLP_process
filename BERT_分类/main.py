#from transformers import AdamW, BertTokenizer, BertModel, BertForMaskedLM, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
import argparse
import torch
import utils
import models.bert_line as bert
import train
#设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义超参数
parser = argparse.ArgumentParser()
#本地
parser.add_argument('--data_path', type=str, default="/Users/ren/Desktop/nlp相关/实验1/aclImdb/")#文件路径
#parser.add_argument('--data_path', type=str, default="/data/renhongjie/zouye1_new/data/aclImdb/")#文件路径
parser.add_argument('--embed_size', type=int, default=300)#embeding层宽度
parser.add_argument('--num_hidens', type=int, default=100)
parser.add_argument('--seq_len', type=int, default=300)#文件长度，需要截断和填充
parser.add_argument('--batch_size', type=int, default=64)#批次
parser.add_argument('--bidirectional', type=bool, default=True)#是否开启双向
parser.add_argument('--num_classes', type=int, default=2)#分类个数
parser.add_argument('--lr', type=float, default=1e-4)#学习率
parser.add_argument('--droput', type=float, default=0.5)#丢弃率
parser.add_argument('--num_epochs', type=int, default=10)#训练论数
parser.add_argument('--vocab_size', type=int, default=0)#vocab大小
parser.add_argument('--save_path', type=str, default="best.pth")#保存路径
parser.add_argument('--CLS', type=str, default="[CLS]")#CLS标记
parser.add_argument('--PAD', type=str, default="[PAD]")#PAD标记
parser.add_argument('--weight_decay', type=float, default=1e-4)#权重衰减
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
model = AutoModel.from_pretrained("./bert-base-uncased")

parser.add_argument('--tokenizer', default=tokenizer)#保存路径
parser.add_argument('--bert_model', default=model)#保存路径
parser.add_argument('--hidden_size', type=int, default=768)#看模型配置，768
args = parser.parse_args()
train_iter,test_iter,vail_iter=utils.data(args)
# inputs = args.tokenizer("Hello world!", return_tensors="pt")
# outputs = args.bert_model(**inputs)
# print(outputs)
net=bert.Model(args)
net.to(device)
train.train(args,device,net,train_iter,test_iter,vail_iter)