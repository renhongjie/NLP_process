
# BiLstm-CRF
#### 数据集
[CONLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/)

#### 词向量
使用glove预训练的embedding进行初始化

#### 模型图

![Image text](https://github.com/renhongjie/NLP_process/tree/main/images/BiLSTM&CRF.PNG)


#### 准确率
可以在测试集上达到准确率76%+（感觉有些问题没解决，组内其他人写的可以88%+）
#### 注意事项
代码需要修改数据集路径和词向量路径

（数据集和词向量文件未提供，请自行下载）
#### 存在的问题（未解决）
1、可能和CRF有关系，只使用BiLstm也可以70%+以上效果

2、可以和没有mask有关


### 项目结构描述
```
├── README.md                   // 描述文件
├── main.py                     // 主函数文件/运行文件
├── data_process.py             // 数据处理函数集合
├── model.py                    // 模型文件
├── train.py                    // 训练函数            
├── CRF.py                      // github上下载的别人的CRF         
└── utils.py                    //工具函数集合
```