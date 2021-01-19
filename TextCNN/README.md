# TextCNN
#### 数据集
[aclImdb](http://ai.stanford.edu/~amaas/data/sentiment/)

#### 词向量
使用glove预训练的embedding进行初始化

#### 模型图

![Image text](https://github.com/renhongjie/NLP_process/tree/main/images/TextCNN.PNG)

![Image text](https://github.com/renhongjie/NLP_process/tree/main/images/TextCNN2.png)

#### 准确率
可以在测试集上达到准确率89%+（复习论文，但是没达到论文的最好效果）
#### 注意事项
代码需要修改数据集路径和词向量路径

（数据集和词向量文件未提供，请自行下载）

### 项目结构描述
```
├── README.md                   // 描述文件
├── main.py                     // 主函数文件/运行文件
├── data_process.py             // 数据处理函数集合
├── model.py                    // 模型文件
├── train.py                    // 训练函数                 
└── utils.py                    //工具函数集合
```
