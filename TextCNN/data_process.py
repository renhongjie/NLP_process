import re
from itertools import chain
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

# 去掉停用词+还原词性
# def stop_words(text):
#     stop_words = stopwords.words("english")
#     wl = WordNetLemmatizer()
#     words = wl.lemmatize(text)
#     return words

# 缩略词还原：不过处理贼慢...
replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would')]


class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

replacer = RegexpReplacer()

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
    text = re.sub("[^a-zA-Z']", "", text.lower())
    text = " ".join([word for word in text.split(' ')])
    return text

def get_token_text(text):
    token_data = [data_process(st) for st in text.split()]
    # token_data = [st.lower() for st in text.split()]
    token_data = list(filter(None, token_data))
    return token_data


def get_token_data(data):
    data_token = []
    for st, label in data:
        data_token.append(get_token_text(st))
    return data_token


# 建立词典
def get_vocab(data):
    vocab = set(chain(*data))
    vocab_size = len(vocab)
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'
    return vocab, vocab_size, word_to_idx, idx_to_word

# 转化为索引
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
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while (len(padded_feature) < maxlen):
                padded_feature.append(pad)
        padded_features.append(padded_feature)
    return padded_features

# 处理验证集的
def read_file(data_path):
    data = []
    for line in open(data_path):
        token_data = [data_process(st) for st in line.split()]
        token_data = list(filter(None, token_data))
        data.append(token_data)
    return data