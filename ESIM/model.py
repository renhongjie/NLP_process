import torch
import torch.nn as nn
import torch.nn.functional as F

class ESIM(nn.Module):
    def __init__(self, args, weight):
        super(ESIM, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size+1, args.embed_size)
        self.embedding.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding.weight.requires_grad = True
        # self.bn_embeds = nn.BatchNorm1d(self.embedding)
        self.lstm1 = nn.LSTM(args.embed_size, args.hidden_size, batch_first=True, bidirectional=args.bidirectional)
        self.lstm2 = nn.LSTM(args.hidden_size * 8, args.hidden_size, batch_first=True, bidirectional=args.bidirectional)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(args.hidden_size * 8),
            nn.Linear(args.hidden_size * 8, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(args.dropout),
            nn.Linear(2, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(args.dropout),
            nn.Linear(2, args.classification_num),
            nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        # 求解eij
        attention = torch.matmul(x1, x2.transpose(1, 2))
        # print("attention.shape:",attention.shape)
        # mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        # mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        # weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        # weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # print("weight1.shape:",weight1.shape)
        # print("weight2.shape:",weight2.shape)
        # print("x1_align.shape:",x1_align.shape)
        # print("x2_align.shape:",x2_align.shape)
        # x_align: batch_size * seq_len * hidden_size
        # 返回波浪形a，b
        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, x1, x2):
        # batch_size * seq_len
        sent1, sent2 = x1, x2

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        # x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        # x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        x1 = self.embedding(sent1)
        # print("x1.shape:",x1.shape)
        x2 = self.embedding(sent2)
        # print("x2.shape:",x2.shape)
        # batch_size * seq_len * dim =>      batch_size * seq_len * hidden_size
        # 杠a,b
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)
        # print("o1.shape:",o1.shape)
        # print("o1.shape:",o2.shape)
        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2)
        # print("q1_align.shape:",q1_align.shape)
        # print("q2_align.shape:",q2_align.shape)
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)
        # print("q1_combined.shape:",q1_combined.shape)
        # print("q2_combined.shape:",q2_combined.shape)
        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
        # print("q1_compose.shape:",q1_compose.shape)
        # print("q2_compose.shape:",q2_compose.shape)
        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)
        # print("q1_rep.shape:",q1_rep.shape)
        # print("q2_rep.shape:",q2_rep.shape)
        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity