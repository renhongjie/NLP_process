import data_process
import torch
import gensim

def data(args):
    train_data = data_process.read_files(args.data_path, "train")
    test_data = data_process.read_files(args.data_path, "test")
    train_feature,train_mask,train_label,train_seq_len=data_process.get_data_label(args,train_data)
    test_feature,test_mask, test_label ,test_seq_len= data_process.get_data_label(args,test_data)
    vail_feature, vail_mask, vail_seq_len=data_process.read_vail(args,"test.txt")
    #print(train_feature[0:2],train_mask[0:2],train_seq_len[0:2],train_label[0:2])
    #print(test_feature[0:2],test_mask[0:2],test_seq_len[0:2], test_label[0:2])
    #转化为tensor
    train_feature, train_mask, train_label, train_seq_len=data_process.to_tensor(train_feature, train_mask, train_label, train_seq_len)
    test_feature, test_mask, test_label, test_seq_len=data_process.to_tensor(test_feature,test_mask, test_label ,test_seq_len)
    train_set = torch.utils.data.TensorDataset(train_feature,train_mask,train_label,train_seq_len)
    test_set=torch.utils.data.TensorDataset(test_feature, test_mask, test_label, test_seq_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    vail_feature, vail_mask, _,vail_seq_len=data_process.to_tensor(vail_feature, vail_mask, vail_mask,vail_seq_len)
    vail_set = torch.utils.data.TensorDataset(vail_feature, vail_mask, vail_seq_len)
    vail_iter = torch.utils.data.DataLoader(vail_set, batch_size=args.batch_size, shuffle=False)
    return train_iter,test_iter,vail_iter