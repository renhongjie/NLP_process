# 词云展示
成果展示：
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/词云1.png"/></div>
<p align="center">图1</p>
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/词云2.png"/></div>
<p align="center">图2</p>
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/词云3.png"/></div>
<p align="center">图3</p>
<div align=center><img  src="https://github.com/renhongjie/NLP_process/blob/main/images/词云4.png"/></div>
<p align="center">图4</p>

# NLP相关流程：
分词->统计词频->建立词云

建立词云貌似有两种方式：一种是直接传入文本，另一种传入词和对应的出现次数（本项目使用第二种方式）
### 项目结构描述
```
├── README.md       // 描述文件
├── 中文词云.py     // 主函数文件/运行文件
├── 英文词云.py     // 主函数文件/运行文件
├── 改变背景.py     // 主函数文件/运行文件
├── 中文.txt        // 中文的数据文本 
├── xin.jpeg        // 所用背景图 
├── man.jpeg        // 所用背景图 
├── 哈工大停用词.txt// 哈工大停用词+自己添加部分停用词，好像没用上？
└── 英文.txt        //英文的数据文本
```

## 英文词云：
### 1、分词：大多数情况下以空格进行分割（本项目用的jieba）
### 2、处理停用词：本项目未处理
### 3、设计vocab：统计词频、排序
### 4、建立词云


## 中文文本：
### 1、分词：比英文复杂一点，往往采用jieba分词等工具进行分词（本项目使用jieba）
### 2、处理分词：相对于英语该部分比较少
### 3、设计vocab：统计词频、排序
### 4、建立词云
（其实没啥区别，不过建立中文词云需要设置字体，否则会是框）

## 文件介绍：
### 中文词云.py：最基本的词云（图1）
### 英文词云.py：最基本的词云（图2）
### 改变背景.py：更改形状和字体颜色的词云（图3、图4）

# NLP相关流程：
分词->统计词频->建立词云

建立词云貌似有两种方式：一种是直接传入文本，另一种传入词和对应的出现次数（本项目使用第二种方式）

## 英文词云：
### 1、分词：大多数情况下以空格进行分割（本项目用的jieba）
### 2、处理停用词：本项目未处理
### 3、设计vocab：统计词频、排序
### 4、建立词云


## 中文文本：
### 1、分词：比英文复杂一点，往往采用jieba分词等工具进行分词（本项目使用jieba）
### 2、处理分词：相对于英语该部分比较少
### 3、设计vocab：统计词频、排序
### 4、建立词云
（其实没啥区别，不过建立中文词云需要设置字体，否则会是框）