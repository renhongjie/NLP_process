import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
import os
def train(args,device,net,train_iter,test_iter,vail_iter):
    best_test_acc = 0
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    net=net.to(device)
    print("模型开始训练")
    for epoch in range(args.num_epochs):
        start = time.time()
        train_loss, test_losses = 0, 0
        train_acc, test_acc = 0, 0
        n, m = 0, 0
        net.train()
        for feature, label in train_iter:
            n += 1
            feature = feature.transpose(1, 0)
            feature = feature.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            score = net(feature[0], feature[1])
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
            train_loss += loss
        with torch.no_grad():
            net.eval()
            for test_feature, test_label in test_iter:
                m += 1
                test_feature = test_feature.transpose(1, 0)
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_score = net(test_feature[0], test_feature[1])

                test_loss = loss_function(test_score, test_label)
                test_acc += accuracy_score(torch.argmax(test_score.cpu().data, dim=1), test_label.cpu())
                test_losses += test_loss
        end = time.time()
        runtime = end - start
        print(
            'epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best test acc: %.5f,time: %.4f \n' % (
                epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, best_test_acc / m,
                runtime))
        if best_test_acc/m < test_acc / m and test_acc / m > 0.1:
            best_test_acc = test_acc
            #torch.save(net, args.save_path)
            if args.if_vail:
                if os.path.exists("result.txt"):  # 如果文件存在
                    # 删除文件，可使用以下两种方法。
                    os.remove("result.txt")
                    # os.unlink(path)
                for feature in vail_iter:
                    f = open("result.txt","a+")
                    feature = feature[0].transpose(1, 0)
                    feature = feature.to(device)
                    score = net(feature[0], feature[1])
                    f.write(str(torch.argmax(score.cpu().data, dim=1)))
                    f.close()
