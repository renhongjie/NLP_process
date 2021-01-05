from py2neo import Graph

class AnswerSearcher:
    def __init__(self):
        self.g = Graph("http://localhost:7474", username="neo4j", password="rhjhss")
        self.num_limit = 20

    '''执行cypher查询，并返回相应结果'''
    def search_main(self, sqls):
        final_answers = []
        for sql_ in sqls:
            question_type = sql_['question_type']
            queries = sql_['sql']
            answers = []
            for query in queries:
                ress = self.g.run(query).data()
                answers += ress
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    '''根据对应的qustion_type，调用相应的回复模板'''
    def answer_prettify(self, question_type, answers):
        final_answer = []
        if not answers:
            return ''
        #十面埋伏和功夫的评分（测试完成，单个和多个）
        #可以完成多个电影查询评分，取第一个评分，不知道为啥返回好多评分。。。
        if question_type == 'pingfen':
            l_=[]
            for i in answers:
                if i['m.title'] not in l_:
                    l_.append(i['m.title'])
                    final_answer = '{0}的评分是：{1}'.format(i['m.title'], i['m.rating'])
                    print(final_answer)
        #十面埋伏和功夫的上映时间（测试完成，单个和多个）
        elif question_type == 'shangying':
            l_ = []
            for i in answers:
                if i['m.title'] not in l_:
                    l_.append(i['m.title'])
                    final_answer = '{0}的上映时间是：{1}'.format(i['m.title'], i['m.releasedate'])
                    print(final_answer)
        #十面埋伏和功夫的风格（测试完成，单个和多个）
        elif question_type == 'fengge':
            dict_ = {}
            #print(answers)
            for i in answers:
                if i['m.title'] not in dict_:
                    dict_[i['m.title']]=i['b.name']
                else:
                    dict_[i['m.title']] += ("、"+i['b.name'])
            #print(dict_)
            for i in dict_:
                print("{0}的类型是：{1}".format(i,dict_[i]))
        #十面埋伏和功夫的简介（测试完成，单个和多个）
        elif question_type == 'jvqing':
            l_ = []
            for i in answers:
                if i['m.title'] not in l_:
                    l_.append(i['m.title'])
                    final_answer = '{0}的剧情是：{1}'.format(i['m.title'], i['m.introduction'])
                    print(final_answer)
        #十面埋伏和功夫的演员（测试完成，单个和多个）
        elif question_type == 'chuyan':
            dict_ = {}
            #print(answers)
            for i in answers:
                if i['m.title'] not in dict_:
                    dict_[i['m.title']] = i['n.name']
                else:
                    dict_[i['m.title']] += ("、" + i['n.name'])
            #print(dict_)
            for i in dict_:
                print("{0}的演员名单是：{1}".format(i, dict_[i]))
        #李连杰和成龙的简介（测试完成，单个和多个）
        elif question_type == 'yanyuanjianjie':
            l_ = []
            #print(answers)
            for i in answers:
                if i['n.name'] not in l_:
                    l_.append(i['n.name'])
                    #添加找不到的处理
                    if i['n.biography']!='':
                        final_answer = '{0}的介绍是：{1}'.format(i['n.name'], i['n.biography'])
                        print(final_answer)
                    else:
                        print("找不到{0}的介绍".format(i['n.name']))

        #成龙和李连杰和周星驰合作的电影（多人测试完成）
        elif question_type == 'hezuochuyan':
            dict_ = {}
            # 构建一个总集合
            l_ = []
            #print(answers)
            for i in answers:
                if i['m.title'] not in l_ :
                    l_.append(i['m.title'])
                if i['n.name'] not in dict_:
                    dict_[i['n.name']] = []
                    dict_[i['n.name']].append(i['m.title'])
                else:
                    dict_[i['n.name']].append(i['m.title'])
            #print(dict_)
            #输出这些人各自的电影
            # for i in dict_:
            #     print("{0}演过的电影有：{1}".format(i, dict_[i]))
            #取交集
            name=''
            for i in dict_:
                name+=(i+"、")
                l_ = list(set(l_).intersection(set(dict_[i])))
            #list转str
            s=''
            for i in l_:
                s+=(i+'、')

            if s=='':
                print("{0}没有共同出演的电影有：{1}".format(name[:-1]))
            else:
                # -1过滤最后一个顿号
                print("{0}共同出演的电影有：{1}".format(name[:-1],s[:-1]))
        #成龙和李连杰和周星驰总共的电影
        elif question_type == 'zonggong':
            #不展示具体有哪些了哈
            dict_ = {}
            #print(answers)
            for i in answers:
                if i['n.name'] not in dict_:
                    dict_[i['n.name']] = []
                    dict_[i['n.name']].append(i['m.title'])
                else:
                    dict_[i['n.name']].append(i['m.title'])
            for i in dict_:
                print("{0}总共演过的电影有：{1}部".format(i, len(dict_[i])))

        #周星驰和李连杰的生日？
        elif question_type == 'shengri':
            l_ = []
            for i in answers:
                if i['n.name'] not in l_:
                    l_.append(i['n.name'])
                    final_answer = '{0}的生日是：{1}'.format(i['n.name'], i['n.birth'])
                    print(final_answer)
        return final_answer


if __name__ == '__main__':
    searcher = AnswerSearcher()