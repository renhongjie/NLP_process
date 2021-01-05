import csv
with open('/genre.csv', 'r', encoding='utf-8') as f:
    #数据集除了第一行代表属性外，第一列为实体1，第二列为实体2，第三列是两者英文关系，第四列为两者中文关系
    l_genre=[]
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        #print("当前行数：",reader.line_num,"当前内容",item)
        #只要类别
        print("当前行数：", reader.line_num, "当前内容", item[1])
        if item[1] not in l_genre:
            l_genre.append(item[1])
with open('/movie.csv', 'r', encoding='utf-8') as f:
    l_movie = []
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        #print("当前行数：",reader.line_num,"当前内容",item)
        #只要电影名字
        print("当前行数：", reader.line_num, "当前内容", item[1])
        if item[1] not in l_movie:
            l_movie.append(item[1])
with open('/person.csv', 'r', encoding='utf-8') as f:
    l_person=[]
    #数据集除了第一行代表属性外，第一列为实体1，第二列为实体2，第三列是两者英文关系，第四列为两者中文关系
    reader=csv.reader(f)
    for item in reader:
        #第一行的标签不是咱们需要的内容，line_num表示文件的第几行
        if reader.line_num==1:
            continue
        #print("当前行数：",reader.line_num,"当前内容",item)
        #只要演员
        print("当前行数：", reader.line_num, "当前内容", item[3])
        if item[3] not in l_person:
            l_person.append(item[3])


f_genre = open('genre.txt', 'w+')
f_genre.write('\n'.join(list(l_genre)))
f_genre.close()

f_movie = open('movie.txt', 'w+')
f_movie.write('\n'.join(list(l_movie)))
f_movie.close()

f_person= open('person.txt', 'w+')
f_person.write('\n'.join(list(l_person)))
f_person.close()