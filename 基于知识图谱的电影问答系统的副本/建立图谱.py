import csv
from py2neo import Graph,Node,Relationship,NodeMatcher,NodeMatcher
import py2neo
#graph直接写账号密码会不安全
g=Graph('http://localhost:7474',user='neo4j',password='rhjhss')
#创建config以及db.cfg用来存储信息


with open('/Users/ren/PycharmProjects/基于知识图谱的电影问答系统/data/genre.csv','r',encoding='utf-8') as f:
    #数据集除了第一行代表属性外，第一列为实体1，第二列为实体2，第三列是两者英文关系，第四列为两者中文关系
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        print("当前行数：",reader.line_num,"当前内容",item)
        test_node_1=Node("Genre",id=item[0],name=item[1])
        g.merge(test_node_1, "Genre", "id")
with open('/Users/ren/PycharmProjects/基于知识图谱的电影问答系统/data/movie.csv','r',encoding='utf-8') as f:
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        print("当前行数：",reader.line_num,"当前内容",item)
        test_node_1=Node("Movie",id=item[0],title=item[1],introduction=item[2],rating=item[3],releasedate=item[4])
        g.merge(test_node_1, "Movie", "id")
with open('/Users/ren/PycharmProjects/基于知识图谱的电影问答系统/data/person.csv','r',encoding='utf-8') as f:
    #数据集除了第一行代表属性外，第一列为实体1，第二列为实体2，第三列是两者英文关系，第四列为两者中文关系
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        print("当前行数：",reader.line_num,"当前内容",item)
        test_node_1=Node("Person",id=item[0],birth=item[1],death=item[2],name=item[3],biography=item[4])
        g.merge(test_node_1, "Person", "id")
matcher = NodeMatcher(g)
findnode = matcher.match('Person', id='9550').first()
print(findnode)
with open('/Users/ren/PycharmProjects/基于知识图谱的电影问答系统/data/person_to_movie.csv','r',encoding='utf-8') as f:
    #数据集除了第一行代表属性外，第一列为实体1，第二列为实体2，第三列是两者英文关系，第四列为两者中文关系
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        print("当前行数：",reader.line_num,"当前内容",item)
        findnode = matcher.match('Person', id=item[0]).first()
        endnode = matcher.match('Movie', id=item[1]).first()
        relationships = Relationship(findnode, '饰演', endnode)
        g.merge(relationships, "", "id")
with open('/Users/ren/PycharmProjects/基于知识图谱的电影问答系统/data/movie_to_genre.csv', 'r', encoding='utf-8') as f:
    #数据集除了第一行代表属性外，第一列为实体1，第二列为实体2，第三列是两者英文关系，第四列为两者中文关系
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        print("当前行数：",reader.line_num,"当前内容",item)
        findnode = matcher.match('Movie', id=item[0]).first()
        endnode = matcher.match('Genre', id=item[1]).first()
        relationships = Relationship(findnode, '是', endnode)
        g.merge(relationships, "", "id")