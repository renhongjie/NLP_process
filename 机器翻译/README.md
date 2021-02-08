# NLP练习
### 基于Seq2Seq的机器翻译

#### 数据集：都提供了
#### 经过200轮训练大概loss从7点几到了0.5，大概400轮可以到0.1多点
# 部分细节：
### 1、数据预处理并封装成特定格式，可能是batch*seq_len,也可以是seq_len*batch，网上项目后者比较多，我一般习惯前者
### 2、模型训练中50%用预测值当做解码器输输入值，50%用实际值当解码器输入值。测试阶段全部使用预测值
### 3、构建的热力图其实就是用的attenttion权重
### 4、进行预测时，注意处理特殊词（sos、pad、eos等）

# 模型图
![Image text](https://github.com/renhongjie/NLP_process/blob/main/images/seq2seq.png)
图1

### 为了了解编码器、解码器和注意力机制写的，文件结构比较乱乱的，没有整合起来。跑了跑了字符级翻译、词级别翻译、batch*seq_len数据格式、seq_len*batch数据格式、注意力机制等。S2S_词级_small-atten.ipynb应该是比较完整的，建议参考这个


### 项目结构描述
```
├── README.md       // 描述文件
├── data            // 稍大点的中英数据集文件 
│    ├── news-commentary-v13.zh-en.en  //google machine translation英语数据集 
│    └── news-commentary-v13.zh-en.zh  //google machine translation中文数据集 
├── cmn-eng   // 稍小点的中英数据集文件，下载后解压的文件，下载指令文件中有 
│    ├── cmn.txt               // 中英数据集
│    └── _about.txt            // 不知道是啥文件
├── S2S_字符级_small.ipynb     // S2S模型、字符级别、小数据
├── S2S_词级_small.ipynb       // S2S模型、词级别、小数据
├── S2S_词级_small-2.ipynb     // S2S模型、字符级别、小数据
├── S2S_词级_big.ipynb         // S2S模型、字符级别、大数据，没跑太久，一个多小时一轮...顶不举啊
└── S2S_词级_small-atten.ipynb // S2S模型、词级别、小数据、注意力机制
```

### 部分结果图展示
![Image text](https://github.com/renhongjie/NLP_process/blob/main/images/机器翻译结果1.png)

图2

![Image text](https://github.com/renhongjie/NLP_process/blob/main/images/机器翻译结果2.png)

图3

