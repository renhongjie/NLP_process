import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from CRF import CRF
from torch.autograd import Variable
import torch


class NERLSTM_CRF(nn.Module):
    def __init__(self,args, weight):
        super(NERLSTM_CRF, self).__init__()

        # self.embedding_dim = embedding_dim
        self.embed_size = args.embed_size
        self.hidden_dim = args.hidden_dim
        self.vocab_size = args.vocab_size + 1
        self.tagset_size = args.seq_len

        # self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.embedding.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(args.dropout)

        # CRF
        self.lstm = nn.LSTM(self.embed_size, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=False)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size + 2)
        self.crf = CRF(target_size=self.tagset_size, average_batch=True, use_cuda=False)

    def forward(self, x):
        # CRF
        # x = x.transpose(0,1)
        # print(x.shape)
        batch_size = x.size(0)
        sent_len = x.size(1)

        embedding = self.embedding(x)
        # print(embedding.shape)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        # print(outputs.shape)
        # CRF
        # outputs = self.crf(outputs)
        return outputs

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        # print(feats.shape, mask.shape, tags.shape)
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value