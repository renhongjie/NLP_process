import re
from itertools import chain
import os


# 返回数组，分别为文本+特征
def read_file(data_path, label_path):
    data = []
    labels = []
    for line in open(data_path):
        data.append(line.strip().split(' '))
    for line in open(label_path):
        labels.append(line.strip().split(' '))
    return data, labels


def get_stop_words_list(filepath):
    stop_words_list = []
    with open(filepath, encoding='utf8') as f:
        for line in f.readlines():
            stop_words_list.append(line.strip())
    return stop_words_list


# 特殊处理数据+去标点
def data_process(text):
    text = text.lower()
    # 特殊数据处理，该地方参考的殷同学的
    text = text.replace("<br /><br />", "").replace("it's", "it is").replace("i'm", "i am").replace("he's",
                                                                                                    "he is").replace(
        "she's", "she is") \
        .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's",
                                                                                                        "that is") \
        .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't",
                                                                                                     "does not") \
        .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not") \
        .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's",
                                                                                                          "let us") \
        .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't",
                                                                                                        "could not") \
        .replace("you'll", "you will").replace("you've", "you have")
    # 去除标点
    # text = re.sub("[^a-zA-Z']", "", text.lower())
    text = " ".join([word for word in text.split(' ')])
    return text


def get_token_text(text):
    token_data = [data_process(st) for st in text]
    # token_data = [st.lower() for st in text.split()]
    token_data = list(filter(None, token_data))
    return token_data


# 返回文本分词形式
def get_token_data(data):
    data_token = []
    for st in data:
        data_token.append(get_token_text(st))
    return data_token


def get_vocab(data):
    # 将分词放入set，不重复。类似建立语料库
    vocab = set(chain(*data))
    vocab_size = len(vocab)
    # 建立语料库和索引
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'
    return vocab, vocab_size, word_to_idx, idx_to_word


def encode_st(token_data, vocab, word_to_idx):
    features = []
    for sample in token_data:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


# 填充和截断
def pad_st(features, maxlen, pad=0):
    mask = []
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            mask.append([1] * maxlen)
            padded_feature = feature[:maxlen]
        else:
            m = len(feature) * [1]

            padded_feature = feature
            while (len(padded_feature) < maxlen):
                m.append(0)
                padded_feature.append(pad)
            mask.append(m)
        padded_features.append(padded_feature)
    return padded_features, mask