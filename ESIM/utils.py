import data_process
import torch
import gensim
def load_weight(args,word_to_idx,idx_to_word):
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_path, binary=False, encoding='utf-8')
    weight = torch.zeros(args.vocab_size + 1, args.embed_size)
    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
            #print("成功")
        except:
            #print("失败")
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
    return weight
def data(args):
    train_data, train_label = data_process.read_file(args.data_path, 'snli_1.0_train.txt')
    test_data, test_label = data_process.read_file(args.data_path, 'snli_1.0_test.txt')

    print("read_file success!")

    train_token = data_process.get_token_data(train_data)

    test_token = data_process.get_token_data(test_data)

    print("get_token_data success!")

    vocab_t, vocab_size_t, word_to_idx_t, idx_to_word_t = data_process.get_vocab(
        train_token[0] + train_token[1] + test_token[0] + test_token[1])
    # np.save("vocab_t.npy",vocab_t)
    vocab_l, vocab_size_l, word_to_idx_l, idx_to_word_l = data_process.get_vocab([train_label])
    # np.save("vocab.npy",vocab)
    args.vocab_size=vocab_size_t
    print("vocab_save success!")
    print(idx_to_word_l )
    train_features1 = data_process.pad_st(data_process.encode_st(train_token[0], vocab_t, word_to_idx_t), args.seq_len)
    train_features2 = data_process.pad_st(data_process.encode_st(train_token[1], vocab_t, word_to_idx_t), args.seq_len)
    train_features = []
    for i, j in zip(train_features1, train_features2):
        train_features.append([i, j])
    test_features1 = data_process.pad_st(data_process.encode_st(test_token[0], vocab_t, word_to_idx_t), args.seq_len)
    test_features2 = data_process.pad_st(data_process.encode_st(test_token[1], vocab_t, word_to_idx_t), args.seq_len)
    test_features = []
    for i, j in zip(test_features1, test_features2):
        test_features.append([i, j])
    train_labels = (data_process.encode_sl(train_label, vocab_l, word_to_idx_l))
    test_labels = (data_process.encode_sl(test_label, vocab_l, word_to_idx_l))


    train_features = torch.tensor(train_features[:1000])
    test_features = torch.tensor(test_features[:1000])
    train_labels = torch.tensor(train_labels[:1000])
    test_labels = torch.tensor(test_labels[:1000])

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    weight = load_weight(args, word_to_idx_t, idx_to_word_t)
    if args.if_vail:
        vail_data = data_process.read_test("snli.test")

        print(vail_data[0:3])
        vail_token = data_process.get_token_data(vail_data)
        print("vail_token",vail_token[0][0:3],vail_token[1][0:3])
        vail_features1 = data_process.pad_st(data_process.encode_st(vail_token[0], vocab_t, word_to_idx_t), args.seq_len)
        vail_features2 = data_process.pad_st(data_process.encode_st(vail_token[1], vocab_t, word_to_idx_t), args.seq_len)
        print("vail_features1",vail_features1[0:3])
        vail_features=[]
        for i, j in zip(vail_features1, vail_features2):
            vail_features.append((i, j))
        print(vail_features[0:2])
        vail_features = torch.tensor(vail_features)
        vail_set = torch.utils.data.TensorDataset(vail_features)
        vail_iter = torch.utils.data.DataLoader(vail_set, batch_size=args.batch_size, shuffle=False)
        return  train_iter, test_iter,vail_iter,weight
    return train_iter, test_iter, weight