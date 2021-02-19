# NLP练习
### 基于Seq2Seq+指针网络的文本摘要

#### 数据集：https://pan.baidu.com/s/1AKjjxmZMmNa1SMATjtykHg  密码: w6vj 
#### 数据比较多，大概全部数据用8批次要两个多小时一轮，选取大约1/5数据后测试，经过200轮训练大概loss从5点几到了0.3

# 部分细节：

### 1、数据预处理并封装成特定格式，使用seq_len * batch，并且过长的数据过滤掉
### 2、论文参考：https://arxiv.org/pdf/1704.04368.pdf


# 模型图
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/s2s+指针网络.png"/></div>
<p align="center">图1</p>

## 模型详细分析
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/s2s+指针网络资料1.png"/></div>
<p align="center">图2</p>
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/s2s+指针网络资料2.png"/></div>
<p align="center">图3</p>

### main.py是我服务器上跑的文件，ipynb文件是整个编程过程的文件
### 项目参考博文：https://www.cnblogs.com/zingp/p/11571593.html

### 项目结构描述
```
├── README.md                   // 描述文件
├── main.py                     // 服务器上的训练文件，直接加载处理好的结果和模型
└── 中文摘要提取.ipynb            // 实际的全部流程都在这里
```

### 训练过程展示
#### 部分数据
```
epoch: 0, train loss: 5.3064,best_loss:5.3064,time: 328.9234 
epoch: 1, train loss: 4.9474,best_loss:5.3064,time: 321.2385 
epoch: 2, train loss: 4.7487,best_loss:5.3064,time: 321.2805 
epoch: 3, train loss: 4.6442,best_loss:5.3064,time: 321.4631 
epoch: 4, train loss: 4.5082,best_loss:5.3064,time: 321.0871 
epoch: 5, train loss: 4.4096,best_loss:5.3064,time: 320.7415 
epoch: 6, train loss: 4.3185,best_loss:5.3064,time: 321.0748 
epoch: 0, train loss: 5.3072,best_loss:5.3072,time: 327.5531 
epoch: 1, train loss: 4.9369,best_loss:4.9369,time: 330.2142 
epoch: 2, train loss: 4.7620,best_loss:4.7620,time: 329.4604 
epoch: 3, train loss: 4.6118,best_loss:4.6118,time: 329.4486 
epoch: 4, train loss: 4.6411,best_loss:4.6118,time: 321.6772 
epoch: 5, train loss: 4.5728,best_loss:4.5728,time: 329.5210 
epoch: 6, train loss: 4.3833,best_loss:4.3833,time: 329.6541 
epoch: 7, train loss: 4.2389,best_loss:4.2389,time: 330.1869 
epoch: 8, train loss: 4.2481,best_loss:4.2389,time: 320.6874 
epoch: 9, train loss: 4.1492,best_loss:4.1492,time: 328.7182 
epoch: 10, train loss: 4.0903,best_loss:4.0903,time: 328.1435 
epoch: 11, train loss: 3.9804,best_loss:3.9804,time: 328.6105 
epoch: 12, train loss: 3.8827,best_loss:3.8827,time: 328.6531 
epoch: 13, train loss: 3.8143,best_loss:3.8143,time: 328.8958 
epoch: 14, train loss: 3.7355,best_loss:3.7355,time: 328.6280 
epoch: 15, train loss: 3.6742,best_loss:3.6742,time: 328.4749 
epoch: 16, train loss: 3.6509,best_loss:3.6509,time: 328.4152 
epoch: 17, train loss: 3.6362,best_loss:3.6362,time: 328.6702 
epoch: 18, train loss: 3.5504,best_loss:3.5504,time: 330.5708 
epoch: 19, train loss: 3.4576,best_loss:3.4576,time: 329.0075 
epoch: 20, train loss: 3.4167,best_loss:3.4167,time: 328.5628 
epoch: 21, train loss: 3.3577,best_loss:3.3577,time: 329.2213 
epoch: 22, train loss: 3.3221,best_loss:3.3221,time: 329.2347 
epoch: 23, train loss: 3.2820,best_loss:3.2820,time: 328.7701 
epoch: 24, train loss: 3.2897,best_loss:3.2820,time: 321.0409 
epoch: 25, train loss: 3.2858,best_loss:3.2820,time: 320.4389 
epoch: 26, train loss: 3.1473,best_loss:3.1473,time: 328.6465 
epoch: 27, train loss: 3.0482,best_loss:3.0482,time: 328.8939 
epoch: 28, train loss: 2.9933,best_loss:2.9933,time: 328.5966 
...
epoch: 52, train loss: 2.0258,best_loss:2.0258,time: 328.8295 
epoch: 53, train loss: 2.0084,best_loss:2.0084,time: 329.0390 
epoch: 54, train loss: 1.9423,best_loss:1.9423,time: 329.2082 
epoch: 55, train loss: 1.9117,best_loss:1.9117,time: 328.2320 
epoch: 56, train loss: 1.8762,best_loss:1.8762,time: 329.0331 
epoch: 57, train loss: 1.8193,best_loss:1.8193,time: 328.4538 
epoch: 58, train loss: 1.8112,best_loss:1.8112,time: 328.3501 
...
epoch: 71, train loss: 1.4348,best_loss:1.4348,time: 328.6692 
epoch: 72, train loss: 1.4652,best_loss:1.4348,time: 320.6343 
epoch: 73, train loss: 1.4048,best_loss:1.4048,time: 328.3257 
epoch: 74, train loss: 1.3734,best_loss:1.3734,time: 328.3147 
epoch: 75, train loss: 1.3879,best_loss:1.3734,time: 320.3618 
epoch: 76, train loss: 1.3561,best_loss:1.3561,time: 327.9748 
...
epoch: 86, train loss: 1.1841,best_loss:1.1713,time: 321.7559 
epoch: 87, train loss: 1.0838,best_loss:1.0838,time: 330.0040 
epoch: 88, train loss: 1.0518,best_loss:1.0518,time: 330.0092 
epoch: 89, train loss: 1.0617,best_loss:1.0518,time: 321.5709 
epoch: 90, train loss: 1.0595,best_loss:1.0518,time: 321.7141 
epoch: 91, train loss: 1.1075,best_loss:1.0518,time: 322.1315 
epoch: 92, train loss: 1.0086,best_loss:1.0086,time: 329.8312 
epoch: 93, train loss: 1.0158,best_loss:1.0086,time: 321.9503 
epoch: 94, train loss: 1.0546,best_loss:1.0086,time: 321.7278 
epoch: 95, train loss: 0.9564,best_loss:0.9564,time: 329.5176 
...
epoch: 110, train loss: 0.7600,best_loss:0.7334,time: 321.3408 
epoch: 111, train loss: 0.7846,best_loss:0.7334,time: 321.3125 
epoch: 112, train loss: 0.7542,best_loss:0.7334,time: 321.1535 
epoch: 113, train loss: 0.7507,best_loss:0.7334,time: 321.5663 
... 
epoch: 132, train loss: 0.6114,best_loss:0.5949,time: 321.3847 
epoch: 133, train loss: 0.6068,best_loss:0.5949,time: 321.2901 
epoch: 134, train loss: 0.5743,best_loss:0.5743,time: 329.0038 
epoch: 135, train loss: 0.5884,best_loss:0.5743,time: 321.1476 
epoch: 136, train loss: 0.6079,best_loss:0.5743,time: 321.4442 
...
epoch: 151, train loss: 0.5096,best_loss:0.4927,time: 321.3907 
epoch: 152, train loss: 0.4912,best_loss:0.4912,time: 329.3858 
epoch: 153, train loss: 0.4717,best_loss:0.4717,time: 329.2006 
...
epoch: 164, train loss: 0.4371,best_loss:0.4261,time: 321.0580 
epoch: 165, train loss: 0.4295,best_loss:0.4261,time: 321.4788 
epoch: 166, train loss: 0.4370,best_loss:0.4261,time: 321.1896 
epoch: 167, train loss: 0.4531,best_loss:0.4261,time: 321.1798 
...
epoch: 195, train loss: 0.3476,best_loss:0.3272,time: 320.6949 
epoch: 196, train loss: 0.3176,best_loss:0.3176,time: 328.6789 
epoch: 197, train loss: 0.3286,best_loss:0.3176,time: 320.7698 
epoch: 198, train loss: 0.3231,best_loss:0.3176,time: 320.4696 
epoch: 199, train loss: 0.3132,best_loss:0.3132,time: 328.4419 
```


#### 全部数据（在服务器上还没跑完）
```
epoch: 0, train loss: 7.1265,best_loss:7.1265,time: 7640.4316 
epoch: 1, train loss: 6.6073,best_loss:6.6073,time: 7721.3938 
epoch: 2, train loss: 6.3259,best_loss:6.3259,time: 7740.5525 
epoch: 3, train loss: 6.0694,best_loss:6.0694,time: 7747.3212 
epoch: 4, train loss: 5.8326,best_loss:5.8326,time: 7736.6617 
epoch: 5, train loss: 5.6019,best_loss:5.6019,time: 7704.7772 
epoch: 6, train loss: 5.3734,best_loss:5.3734,time: 7819.4765 
epoch: 7, train loss: 5.1480,best_loss:5.1480,time: 7884.9305 
epoch: 8, train loss: 4.9269,best_loss:4.9269,time: 7861.9511 
epoch: 9, train loss: 4.7170,best_loss:4.7170,time: 7962.6236 
epoch: 10, train loss: 4.5161,best_loss:4.5161,time: 7898.7942 
epoch: 11, train loss: 4.3251,best_loss:4.3251,time: 7954.4806 
epoch: 12, train loss: 4.1489,best_loss:4.1489,time: 8037.0350 
epoch: 13, train loss: 3.9809,best_loss:3.9809,time: 8038.6285 
epoch: 14, train loss: 3.8236,best_loss:3.8236,time: 7945.8978 
epoch: 15, train loss: 3.6778,best_loss:3.6778,time: 7677.7541 
epoch: 16, train loss: 3.5395,best_loss:3.5395,time: 7816.5412 
epoch: 17, train loss: 3.4098,best_loss:3.4098,time: 7729.3785 
epoch: 18, train loss: 3.2893,best_loss:3.2893,time: 7726.1474 
epoch: 19, train loss: 3.1753,best_loss:3.1753,time: 7732.6514 
epoch: 20, train loss: 3.0683,best_loss:3.0683,time: 7737.3143 
epoch: 21, train loss: 2.9659,best_loss:2.9659,time: 7734.1358 
epoch: 22, train loss: 2.8719,best_loss:2.8719,time: 7822.1463 
epoch: 23, train loss: 2.7809,best_loss:2.7809,time: 7928.0785 
epoch: 24, train loss: 2.6950,best_loss:2.6950,time: 7916.7895 
epoch: 25, train loss: 2.6169,best_loss:2.6169,time: 7922.2065 
epoch: 26, train loss: 2.5423,best_loss:2.5423,time: 7892.0298 
epoch: 27, train loss: 2.4724,best_loss:2.4724,time: 7767.7198 
epoch: 28, train loss: 2.4077,best_loss:2.4077,time: 8318.3263 
epoch: 29, train loss: 2.3459,best_loss:2.3459,time: 7794.4295 
epoch: 30, train loss: 2.2889,best_loss:2.2889,time: 7870.5218 
epoch: 31, train loss: 2.2350,best_loss:2.2350,time: 7946.3455 
epoch: 32, train loss: 2.1839,best_loss:2.1839,time: 7997.0486 
epoch: 33, train loss: 2.1361,best_loss:2.1361,time: 7909.9956 
epoch: 34, train loss: 2.0919,best_loss:2.0919,time: 7936.1332 
epoch: 35, train loss: 2.0492,best_loss:2.0492,time: 7684.5002 
epoch: 36, train loss: 2.0092,best_loss:2.0092,time: 7703.1325 
epoch: 37, train loss: 1.9704,best_loss:1.9704,time: 7700.5427 
epoch: 38, train loss: 1.9340,best_loss:1.9340,time: 7814.0181 
epoch: 39, train loss: 1.8986,best_loss:1.8986,time: 7689.1215 
epoch: 40, train loss: 1.8655,best_loss:1.8655,time: 7701.4178 
epoch: 41, train loss: 1.8337,best_loss:1.8337,time: 7732.7651 
epoch: 42, train loss: 1.8046,best_loss:1.8046,time: 7965.8920 
epoch: 43, train loss: 1.7754,best_loss:1.7754,time: 8009.2777 
epoch: 44, train loss: 1.7475,best_loss:1.7475,time: 8001.1709 
epoch: 45, train loss: 1.7204,best_loss:1.7204,time: 8057.0033 
epoch: 46, train loss: 1.6947,best_loss:1.6947,time: 8069.6449 
epoch: 47, train loss: 1.6702,best_loss:1.6702,time: 8076.3406 
epoch: 48, train loss: 1.6458,best_loss:1.6458,time: 8033.6718 
epoch: 49, train loss: 1.6231,best_loss:1.6231,time: 8002.8783 
epoch: 50, train loss: 1.6014,best_loss:1.6014,time: 8013.4227 
epoch: 51, train loss: 1.5803,best_loss:1.5803,time: 8024.0852 
epoch: 52, train loss: 1.5593,best_loss:1.5593,time: 8035.3448 
epoch: 53, train loss: 1.5397,best_loss:1.5397,time: 7976.0092 
epoch: 54, train loss: 1.5216,best_loss:1.5216,time: 7775.1256 
epoch: 55, train loss: 1.5030,best_loss:1.5030,time: 7801.6417 
epoch: 56, train loss: 1.4839,best_loss:1.4839,time: 7859.4048 
epoch: 57, train loss: 1.4676,best_loss:1.4676,time: 7820.0682 
epoch: 58, train loss: 1.4519,best_loss:1.4519,time: 7805.5018 
epoch: 59, train loss: 1.4348,best_loss:1.4348,time: 7852.6159 

```


