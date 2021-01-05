from question_classifier import *
from question_parser import *
from answer_search import *

'''问答类'''
class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = '没能理解您的问题，我数据量有限。。。能不能问的标准点'
        res_classify = self.classifier.classify(sent)
        if not res_classify=='':
            print(answer)
        #print('类别：',res_classify)
        res_sql = self.parser.parser_main(res_classify)
        #print('sql语句',res_sql)

        final_answers = self.searcher.search_main(res_sql)
        if final_answers=='':
            print(answer)

            #return '\n'.join(final_answers)

if __name__ == '__main__':
    handler = ChatBotGraph()
    #测试-start
    problems=["十面埋伏和功夫的评分","十面埋伏和功夫的上映时间","十面埋伏和功夫的风格","十面埋伏和功夫的简介","十面埋伏和功夫的演员","李连杰和成龙的简介",
             "成龙和李连杰和周星驰合作的电影","成龙和李连杰和周星驰总共演了多少的电影","成龙和李连杰合作的电影","周星驰和李连杰的生日是？","我女朋友是谁？"]
    for id,problem in enumerate(problems):
        print("第{0}个问题是{1}：".format(id,problem))
        handler.chat_main(problem)
        print("\n")
    print("测试结束")
    #测试-end
    while 1:
        question = input('咨询:')
        handler.chat_main(question)



