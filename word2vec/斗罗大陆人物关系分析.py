import jieba
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
lines=open("/Users/ren/Downloads/斗罗大陆.txt",encoding = 'GB2312')
#加载停用词 一定要加.strip() 这个地方改了好久 一直停用词去不掉，输出后才发现带着\n
stop_word=[]
with open('/Users/ren/Desktop/nlp相关/nlp各种大文件/哈工大停用词.txt', 'r') as f:
     for line in f.readlines():  # 按行读文件
        stop_word.append(line.strip())

#加载一下专有名词，纯手工收集，可能不全或者重复，有兴趣的可以在完善一下
jieba.load_userdict("/Users/ren/Desktop/nlp相关/nlp各种大文件/斗罗大陆专有名词.txt")
#将分词结果存储到list内
l_=[]
for line in lines:
    #print(line)
    l=jieba.cut(line.replace("/n","").replace("/r","").replace("\u3000","").replace("”","").replace("“",""))
    for i in l:
        if i not in stop_word:
            l_.append(i)
#word2vec需要词以及空格隔开的形式～
s=''
for i in l_:
    s+=(" "+i)
#保存下来
fh = open('1.txt', 'w+', encoding='utf-8')
fh.write(s)
fh.close()
sentences=LineSentence('1.txt')
model=Word2Vec(sentences,size=100, hs=1, min_count=1, window=3)

#看两个实体之间的相似度
name1="泰坦巨猿"
name2="天青牛蟒"
try:
    sim1 = model.similarity(name1, name2)
    print('{0} 和 {1} 的相似度为：{2}\n'.format(name1,name2,sim1))
except:
    print('{0} 或 {1} 可能不存在～：\n'.format(name1,name2))

#看某实体最相似的n个实体
name='比比东'
n = 8
try:
    for key in model.similar_by_word(name, topn=100):
        if len(key[0]) == 3:
            n -= 1
            print(key[0], key[1])
            if n == 0:
                break
except:
    print(u'没有在文章中找到 {0} ：\n'.format(name))

#看某实体的相似列表
name='泰坦巨猿'
try:

    sim3 = model.most_similar(name, topn=20)
    print(u'和 {0} 与相关的词有：\n'.format(name))
    for key in sim3:
        print(key[0], key[1])
except:
    print(u'没有在文章中找到 {0} ：\n'.format(name))
#保存模型
save_path="斗罗大陆人物关系.model"
model.save(save_path)
#加载模型
model2 = word2vec.Word2Vec.load(save_path)
#加载模型后的测试
name="玄天功"
sim3 = model2.most_similar(name, topn=20)
print(u'和 {0} 与相关的词有：\n'.format(name))
for key in sim3:
    print(key[0], key[1])