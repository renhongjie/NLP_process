
# BERT文本分类
#### 数据集
[aclImdb](http://ai.stanford.edu/~amaas/data/sentiment/)

#### 词向量
使用glove预训练的embedding进行初始化

# 模型图
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/bert.png"/></div>
<p align="center">图1</p>


### bert使用的bert-base-uncased


#### 准确率
可以在测试集上达到准确率92.8%+（未调参数，bert！nb！随便允许一下就92%+）
#### 注意事项
代码需要修改数据集路径和词向量路径

（数据集和pytorch_model.bin未提供，请自行下载）



### 项目结构描述
```
├── README.md                   // 描述文件
├── main.py                     // 主函数文件/运行文件
├── data_process.py             // 数据处理函数集合
├── model                       // 后续打算写其他bert分类模型
│   ├── bert_line.py            // bert+全连接，最基本的bert分类模型
├── train.py                    // 训练函数  
├── bert-base-uncased           // bert三件套 
│   ├── config.json             // bert的配置文件
│   ├── vocab.txt               // bert的词表
│   ├── pytorch_model.bin       // bert的预训练模型
└── utils.py                    //工具函数集合
```