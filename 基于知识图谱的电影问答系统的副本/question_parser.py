class QuestionPaser:

    '''构建实体节点'''
    def build_entitydict(self, args):
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)

        return entity_dict

    '''解析主函数'''
    def parser_main(self, res_classify):
        #提取出实体
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        #提取出查询类型
        question_types = res_classify['question_types']
        sqls = []
        for question_type in question_types:
            sql_ = {}
            sql_['question_type'] = question_type
            sql = []
            if question_type == 'pingfen':
                sql = self.sql_transfer(question_type, entity_dict.get('movie'))

            elif question_type == 'shangying':
                sql = self.sql_transfer(question_type, entity_dict.get('movie'))

            elif question_type == 'fengge':
                sql = self.sql_transfer(question_type, entity_dict.get('movie'))

            elif question_type == 'jvqing':
                sql = self.sql_transfer(question_type, entity_dict.get('movie'))

            elif question_type == 'chuyan':
                sql = self.sql_transfer(question_type, entity_dict.get('movie'))

            elif question_type == 'yanyuanjianjie':
                sql = self.sql_transfer(question_type, entity_dict.get('person'))

            elif question_type == 'hezuochuyan':
                sql = self.sql_transfer(question_type, entity_dict.get('person'))

            elif question_type == 'zonggong':
                sql = self.sql_transfer(question_type, entity_dict.get('person'))

            elif question_type == 'shengri':
                sql = self.sql_transfer(question_type, entity_dict.get('person'))



            if sql:
                sql_['sql'] = sql

                sqls.append(sql_)

        return sqls

    '''针对不同的问题，分开进行处理'''
    def sql_transfer(self, question_type, entities):
        if not entities:
            return []

        # 查询语句
        sql = []
        # 查询评分
        if question_type == 'pingfen':

            #sql=["match (m:Movie)-[]->() where m.title='{0}' return m.rating,m.title".format(entities[0])]
            #多个电影查询
            sql = ["match (m:Movie)-[]->() where m.title='{0}' return m.rating,m.title".format(i) for i in entities]

        # 查询上映
        elif question_type == 'shangying':
            #sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.prevent".format(i) for i in entities]
            sql=["match(m:Movie)-[]->() where m.title='{0}' return m.releasedate,m.title".format(i) for i in entities]
        # 风格
        elif question_type == 'fengge':
            sql = ["match(m:Movie)-[r:`是`]->(b) where m.title=\"{0}\" return b.name,m.title".format(i) for i in entities]

        # 剧情
        elif question_type == 'jvqing':
            sql = ["match(m:Movie)-[]->() where m.title='{0}' return m.title,m.introduction".format(i) for i in entities]

        # 出演
        elif question_type == 'chuyan':
            sql = ["match(n:Person)-[r:`饰演`]->(m:Movie) where m.title=\"{0}\" return m.title,n.name".format(i) for i in entities]

        # 演员介绍
        elif question_type == 'yanyuanjianjie':
            sql = ["match(n:Person)-[]->() where n.name=\"{0}\" return n.name,n.biography".format(i) for i in entities]

        # 合作出演
        elif question_type == 'hezuochuyan':
            sql = ["match(n:Person)-[r:`饰演`]->(m:Movie) where n.name=\"{0}\" return m.title,n.name".format(i) for i in entities]

        # 总共
        elif question_type == 'zonggong':
            sql = ["match(n:Person)-[r:`饰演`]->(m:Movie) where n.name=\"{0}\" return m.title,n.name".format(i) for i in entities]

        # 生日
        elif question_type == 'shengri':
            sql = ["match(n:Person)-[]->() where n.name='{0}' return n.birth,n.name".format(i) for i in entities]


        return sql



if __name__ == '__main__':
    handler = QuestionPaser()
