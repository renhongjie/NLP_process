import jieba    #分词包
import matplotlib.pyplot as plt #matplotlib作为常用的可视化工具
import matplotlib
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#参考博文：https://blog.csdn.net/fly910905/article/details/77763086/
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
jieba.add_word('哑舍')
#停用词
stop_word=[]
with open('哈工大停用词.txt', 'r') as f:
     for line in f.readlines():  # 按行读文件
        stop_word.append(line)
stop_word.append("没有")
stop_word.append("自己")
stop_word.append("一个")
stop_word.append("知道")
stop_word.append("什么")
stop_word.append("已经")
data=[]
with open('哑舍1.txt', 'r') as f:
     for line in f.readlines():  # 按行读文件
          #jieba分词
          line=jieba.cut(line)
          for i in line:
               #去掉停用词和短词
               if i not in stop_word and len(i)>1:
                    data.append(i)

#print(Counter(data))
#统计词频并排序
top_1000 = Counter(data).most_common(1000)
# data=[]
# for i in top_1000:
#      #print(i)
#      data.append(i[0])
#backgroud_Image = plt.imread('man.jpeg')
backgroud_Image = plt.imread('xin.jpeg')
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0) # 设定图像尺寸
wordcloud=WordCloud(font_path="/System/Library/Fonts/STHeiti Light.ttc",background_color="white",max_font_size=80) # 设定词云的字体路径，防止中文出错
wordcloud = WordCloud(
    background_color='white',# 设置背景颜色
    mask=backgroud_Image,# 设置背景图片
    font_path='/System/Library/Fonts/STHeiti Light.ttc',  # 若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
    max_words=2000, # 设置最大现实的字数
    stopwords=STOPWORDS,# 设置停用词
    max_font_size=150,# 设置字体最大值
    random_state=30# 设置有多少种随机生成状态，即有多少种配色方案
)
img_colors = ImageColorGenerator(backgroud_Image)
print(img_colors)

word_frequence = {x[0]:x[1] for x in top_1000} # 取前1000个词频最高的词语
print(word_frequence)
wordcloud=wordcloud.fit_words(word_frequence)
#使用后词云变成图色
wordcloud.recolor(color_func=img_colors)
plt.imshow(wordcloud)
plt.show()

