import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
def acc_mask(x,y,mask):
    batch=mask.size(0)
    acc=0
    for i,j,m in zip(x,y,mask):
        #print("i,j",i,j)
        #print(m)
        m=m.cpu()
        i=i.cpu()
        j=j.cpu()
        m=m.numpy().tolist()
        m.reverse()
        #print(m)
        l=len(m)-m.index(1)
        #print(l)
        acc+=accuracy_score(i[0:l],j[0:l])
    return acc/batch
def train(args,device,net,train_iter,test_iter):
    best_test_acc = 0
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("模型开始训练")
    for epoch in range(args.num_epochs):
        start = time.time()
        train_loss, test_loss = 0, 0
        train_acc, test_acc = 0, 0
        n, m = 0, 0
        net.train()
        for x, y, mask in train_iter:
            pred = []
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            n += 1
            feats = net(x)
            feats = feats.to(device)
            path_score, best_path = net.crf(feats, mask.bool())
            pred.extend([t for t in best_path])
            # print(feats)
            loss = net.loss(feats, mask, y)
            loss.backward()
            optimizer.step()
            # print(pred[0:2])
            # print(y[0:2])
            acc = acc_mask(pred, y, mask)
            train_loss += loss
            train_acc += acc
        with torch.no_grad():
            net.eval()
            for x, y, mask in test_iter:
                pred = []
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                m += 1
                feats = net(x)
                feats = feats.to(device)
                path_score, best_path = net.crf(feats, mask.bool())
                pred.extend([t for t in best_path])
                # print(feats)
                loss = net.loss(feats, mask, y)
                loss.backward()
                optimizer.step()
                # print(pred[0:2])
                # print(y[0:2])
                acc = acc_mask(pred, y, mask)
                test_loss += loss
                test_acc += acc

        end = time.time()
        runtime = end - start
        print(
            'epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best test acc: %.5f,time: %.4f \n' % (
                epoch, train_loss.data / n, train_acc / n, test_loss.data / m, test_acc / m, best_test_acc / m,
                runtime))
        if best_test_acc<test_acc / m and test_acc / m>0.8:
            best_test_acc=test_acc / m
            torch.save(net, args.save_path)

