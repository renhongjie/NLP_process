import argparse
import torch
import utils
import model
import train
import test
#设置设备
def test(args,device,net,train_iter):
    net.load_state_dict(torch.load(args.save_path))
    print(net)
    net.eval()
    for feature,_ in train_iter:
        f=open("result.txt")
        feature = feature.transpose(1, 0)
        feature = feature.to(device)
        score = net(feature[0], feature[1])
        f.write(torch.argmax(score.cpu().data, dim=1))
        f.close()