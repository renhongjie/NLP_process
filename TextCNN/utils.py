import data_process
import torch
import gensim

def load_weight(args,word_to_idx,idx_to_word):
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_path, binary=False, encoding='utf-8')
    weight = torch.zeros(args.vocab_size + 1, args.embed_size)
    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
            print("成功")
        except:
            print("失败")
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
    return weight
def data(args):
    train_data = data_process.read_files(args.data_path, "train")
    test_data = data_process.read_files(args.data_path, "test")


    print("read_file success!")

    train_token = data_process.get_token_data(train_data)
    test_token = data_process.get_token_data(test_data)

    print("get_token_data success!")

    vocab, vocab_size, word_to_idx, idx_to_word = data_process.get_vocab(train_token)
    # np.save("vocab.npy",vocab)
    args.vocab_size=vocab_size
    print("vocab_save success!")

    train_features = data_process.pad_st(data_process.encode_st(train_token, vocab, word_to_idx), args.seq_len)
    test_features = data_process.pad_st(data_process.encode_st(test_token, vocab, word_to_idx), args.seq_len)

    train_label = [score for _, score in train_data]
    test_label = [score for _, score in test_data]
    train_features = torch.tensor(train_features)
    test_features = torch.tensor(test_features)
    train_labels = torch.tensor(train_label)
    test_labels = torch.tensor(test_label)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    weight = load_weight(args, word_to_idx, idx_to_word)
    if args.if_vail==True:
        vail_data = data_process.read_file('test.txt')
        vail_data = data_process.pad_st(data_process.encode_st(vail_data, vocab, word_to_idx), args.seq_len)
        vail_features = torch.tensor(vail_data)
        vail_set = torch.utils.data.TensorDataset(vail_features, test_labels)
        vail_iter = torch.utils.data.DataLoader(vail_set, batch_size=args.batch_size, shuffle=False)
        return train_iter,test_iter,vail_iter,weight
    return train_iter, test_iter, weight
