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
    train_data, train_label = data_process.read_file(args.data_path + "train/seq.in", args.data_path + "train/seq.out")
    test_data, test_label = data_process.read_file(args.data_path + "test/seq.in", args.data_path + "test/seq.out")
    print("read_file success!")

    train_token = data_process.get_token_data(train_data)
    test_token = data_process.get_token_data(test_data)
    print("get_token_data success!")

    vocab_t, vocab_size_t, word_to_idx_t, idx_to_word_t = data_process.get_vocab(train_token)
    # np.save("vocab_t.npy",vocab_t)
    vocab_l, vocab_size_l, word_to_idx_l, idx_to_word_l = data_process.get_vocab(train_label)
    # np.save("vocab.npy",vocab)
    args.vocab_size=vocab_size_t
    print("vocab_save success!")

    train_features, mask_train = data_process.pad_st(data_process.encode_st(train_token, vocab_t, word_to_idx_t), args.seq_len)
    test_features, mask_test = data_process.pad_st(data_process.encode_st(test_token, vocab_t, word_to_idx_t), args.seq_len)
    train_label, _ = data_process.pad_st(data_process.encode_st(train_label, vocab_l, word_to_idx_l), args.seq_len)
    test_label, _ = data_process.pad_st(data_process.encode_st(test_label, vocab_l, word_to_idx_l), args.seq_len)


    train_features = torch.tensor(train_features)
    test_features = torch.tensor(test_features)
    train_labels = torch.tensor(train_label)
    test_labels = torch.tensor(test_label)
    train_mask = torch.tensor(mask_train)
    test_mask = torch.tensor(mask_test)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels,train_mask)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels,test_mask)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    weight = load_weight(args, word_to_idx_t, idx_to_word_t)
    return train_iter, test_iter, weight
