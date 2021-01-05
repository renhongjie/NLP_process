import jieba    #分词包
import matplotlib.pyplot as plt #matplotlib作为常用的可视化工具
import matplotlib
from collections import Counter
from wordcloud import WordCloud #词云包
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
jieba.add_word('古北水镇')
#停用词
stop_word=["：","\n","的","那","如","。",",","，"," ","！","~","了","也","是","、","“","”"]
data=[]
with open('中文.txt', 'r') as f:
     for line in f.readlines():  # 按行读文件
          #jieba分词
          line=jieba.cut(line)
          for i in line:
               #去掉停用词
               if i not in stop_word:
                    data.append(i)

#print(Counter(data))
#统计词频并排序
top_1000 = Counter(data).most_common(1000)
# data=[]
# for i in top_1000:
#      #print(i)
#      data.append(i[0])
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0) # 设定图像尺寸
wordcloud=WordCloud(font_path="/System/Library/Fonts/STHeiti Light.ttc",background_color="white",max_font_size=80) # 设定词云的字体路径，防止中文出错
word_frequence = {x[0]:x[1] for x in top_1000} # 取前1000个词频最高的词语
print(word_frequence)
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()

