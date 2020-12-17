import re
from itertools import chain



# 特殊处理数据+去标点
def data_process(text):
    text = text.lower()
    # 特殊数据处理，该地方参考的殷同学的
    text = text.replace("n't", "not")
    # 去除标点
    text = re.sub("[^a-zA-Z']", "", text.lower())
    # 简单空格切分
    text = " ".join([word for word in text.split(' ')])

    # stop_words = stopwords.words("english")
    # wordnet = WordNetLemmatizer()
    # text = " ".join([wordnet.lemmatize(w,'v') for w in text.split(' ') if w not in stop_words])
    text = " ".join([w for w in text.split(' ')])
    return text
def read_file(data_path, name):
    data1 = []
    data2 = []
    labels = []
    flag=0
    for line in open(data_path+name):
        if flag==0:
            flag+=1
            continue
        l=(line.split('	'))
        labels.append(l[0])
        data1.append(l[5])
        data2.append(l[6])
    return [data1 , data2] , labels
def get_token_text(text):
    token_data = [data_process(st) for st in text.split()]
    # token_data = [st.lower() for st in text.split()]
    token_data = list(filter(None, token_data))
    return token_data
#返回文本分词形式
def get_token_data(data):
    data_token1 = []
    data_token2 = []
    for st in data[0]:
        data_token1.append(get_token_text(st))
    for st in data[1]:
        data_token2.append(get_token_text(st))
    return [data_token1,data_token2]
def get_vail_data(data):
    data_token1 = []
    data_token2 = []
    for st in data:
        data_token1.append(get_token_text(st[0]))
        data_token2.append(get_token_text(st[1]))
    return [data_token1,data_token2]
def get_vocab(data):
    #将分词放入set，不重复。类似建立语料库
	vocab = set(chain(*data))
	vocab_size = len(vocab)
    #建立语料库和索引
	word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
	word_to_idx['<unk>'] = 0
	idx_to_word = {i+1: word for i, word in enumerate(vocab)}
	idx_to_word[0] = '<unk>'
	return vocab,vocab_size,word_to_idx,idx_to_word
def encode_st(token_data,vocab,word_to_idx):
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
def encode_sl(token_data,vocab,word_to_idx):
    features = []
    for token in token_data:
        features.append(word_to_idx[token]-1)
    return features
#填充和截断
def pad_st(features,maxlen,pad=0):
	padded_features = []
	for feature in features:
		if len(feature)>maxlen:
			padded_feature = feature[:maxlen]
		else:
			padded_feature = feature
			while(len(padded_feature)<maxlen):
				padded_feature.append(pad)
		padded_features.append(padded_feature)
	return padded_features
def read_test(path):
    data1 = []
    data2 = []
    for line in open(path):
        x1,x2=line.split("|||")
        data1.append(x1)
        data2.append(x2)
    return [data1,data2]






