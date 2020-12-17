import re
from itertools import chain
import torch
import os
# 返回数组，分别为文本+特征
def read_files(path, filetype):
    file_list = []
    pos_path = path + filetype + "/pos/"
    neg_path = path + filetype + "/neg/"
    for f in os.listdir(pos_path):
        file_list += [[pos_path + f, 1]]
    for f in os.listdir(neg_path):
        file_list += [[neg_path + f, 0]]
    data = []
    for fi, label in file_list:
        with open(fi, encoding='utf8') as fi:
            data += [[" ".join(fi.readlines()), label]]
    return data
def read_vail(args,path):
    f=open(path)
    features = []
    masks = []
    seq_lens = []
    for seg in f:
        token = args.tokenizer.tokenize(seg)
        # bert前面需要加一个标识位[CLS]
        token = [args.CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = args.tokenizer.convert_tokens_to_ids(token)

        # 数据填充
        pad_size = args.seq_len
        if pad_size:
            if len(token) < pad_size:
                # 未补充之前进行记录
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids = token_ids + ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        features.append(token_ids)
        masks.append(mask)
        seq_lens.append(seq_len)
    return features, masks, seq_lens

def get_data_label(args,text):
    features=[]
    labels=[]
    masks=[]
    seq_lens=[]
    for seg in text:
        token=args.tokenizer.tokenize(seg[0])
        # bert前面需要加一个标识位[CLS]
        token = [args.CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = args.tokenizer.convert_tokens_to_ids(token)

        # 数据填充
        pad_size = args.seq_len
        if pad_size:
            if len(token) < pad_size:
                # 未补充之前进行记录
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids = token_ids + ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        features.append(token_ids)
        masks.append(mask)
        labels.append(seg[1])
        seq_lens.append(seq_len)
    return features,masks,labels,seq_lens
def to_tensor(x1,x2,x3,x4):
    return torch.tensor(x1),torch.tensor(x2),torch.tensor(x3),torch.tensor(x4)