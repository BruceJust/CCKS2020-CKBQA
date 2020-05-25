# -*- coding: utf-8 -*-
# @Time    : 2020/5/10 21:58
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : main.py
# @Software: PyCharm

from mention_extrator import MentionExtractor, Bert_scoring
from kb import  (
    GetRelationPathsSingle,
    GetEntity2SingleRelation,
    GetEntity2TwoRelation,
    GetTwoEntityRelation,
    GetEntity2Tuple,
    GetRelationPaths,
    GetRelationPaths3,
)
import pickle
import time
from data_process import get_sim_example_features_eval

# train_file = 'corpus\\train_data.pkl'
# train_data = pickle.load(open(train_file, 'rb'))
# train_questions = [train_data[i]['question'] for i in range(len(train_data))]

extractor = MentionExtractor()
scoring = Bert_scoring()



def GetAmnswer(question):
    start1 = time.time()
    mentions_ner = extractor.get_ner_entity(question)
    print(time.time() - start1)
    mentions_ = extractor.extract_mentions(question)
    print(time.time() - start1)
    ans = ''
    if len(mentions_ner) == 0:
        ans = ''
        print('No entity recognized......')
        return ans
    if len(mentions_ner) <= 1:

        # # 添加同名候选mention
        mentions_list = mentions_ner
        mentions = []
        for item in mentions_list:
            mentions.append(item)
            if item in extractor.mention_dic:
                for mention in extractor.mention_dic[item]:
                    mentions.append(mention)

        # 删除于原文毫无关系的mention, 可能会去掉一些别名，所以取消
        # for mention in mentions:
        #     if not overlap(question, mention):
        #         mentions.remove(mention)
        mentions = list(set(mentions))
        # 单entity (1/2跳关系)
        start = time.time()
        relations = [GetRelationPaths3(mention) for mention in mentions]
        print('searching relations for', time.time() - start)

        if len(relations) == 0:
            ans = ''
            # print(f'No valida relation for "{mentions_[0]}"......')
        else:
            # 先去掉relation 与 question 没有一字交叉的
            # pairs = []
            # for i, mention in enumerate(mentions):
            #     for relation in relations[i]:
            #         if relation:
            #             if None not in relation:  # 个别 relation name是 None
            #                 try:
            #                     if overlap(question, relation):
            #                         pairs.append([[mentions[i]], relation])
            #                 except:
            #                     print(question, mentions[i], relation)
            pairs = []
            for i, mention in enumerate(mentions):
                for relation in relations[i]:
                    if relation:
                        if None not in relation:  # 个别 relation name是 None
                            pairs.append([[mentions[i]], relation])


            # 用BERT计算相似度

            dataset = get_sim_example_features_eval(question, pairs, scoring.tokenizer, 50)
            start = time.time()
            scores = []
            for i in range(len(dataset)):
                res = scoring.sim_model.predict(dataset[i])
                scores.append([res[0][1], question, pairs[i][0], pairs[i][1]])
            # res = scoring.batch_score(dataset)
            print(f'scoring for {len(dataset)} relations with',time.time() - start)
            # scores = []
            # for item in pairs:
            #     score = scoring.score(question, item[0], item[1])
            #     scores.append([score, question, item[0], item[1]])
            max_score = 0
            best_pattern = []
            for item in scores:
                if item[0] > max_score:
                    max_score = item[0]
            for item in scores:
                if item[0] == max_score:
                    best_pattern.append(item)
            if len(best_pattern) == 1:
                if len(best_pattern[0][3]) == 1:
                    ans = GetEntity2SingleRelation(best_pattern[0][2][0], best_pattern[0][3][0])
                else:
                    ans =  GetEntity2TwoRelation(best_pattern[0][2][0], best_pattern[0][3][0], best_pattern[0][3][1])
            # 如果有两个候选relation一样的分数，优先选一个relation的
            elif len(best_pattern) == 2:
                for item in best_pattern:
                    if len(item[3]) == 1:
                        ans = GetEntity2SingleRelation(item[2][0], item[3][0])
            else:
                ans =  ''
    elif len(mentions_ner) == 2:
        # 两个entity

        relations = GetTwoEntityRelation(mentions_ner[0], mentions_ner[1])
        if len(relations) == 0:
            ans = ''
        else:
            scores_two = []
            for relation in relations:
                if relation:
                    score = scoring.score(question, mentions_ner, relation)
                    scores_two.append([score, question, mentions_ner, relation])
            max_score = 0
            best_pattern = []
            for item in scores_two:
                if item[0] > max_score:
                    max_score = item[0]
            for item in scores_two:
                if item[0] == max_score:
                    best_pattern = item
            if len(best_pattern) > 0:
                if len(best_pattern[3]) == 1:
                    ans = GetEntity2Tuple(best_pattern[2][0], best_pattern[3][0], best_pattern[3][1] ,best_pattern[2][1])
            else:
                ans = ''
    else:
        ans = ''

    # 多entity
    # predicates = []
    # history = []
    # for entity_1 in mentions:
    #     for entity_2 in mentions:
    #         if entity_1 != entity_2:
    #             if [entity_2, entity_1] not in history:
    #                 history.append([entity_1, entity_2])
    #                 relation = GetTwoEntityRelation(entity_1, entity_2)
    #                 print(entity_1, entity_2)
    #                 if len(relation) > 0:
    #                     predicates.append(([entity_1, entity_2], relation))
    print('finish with', time.time() - start1)
    return ans


# def Statistics_mention_relations(questions):
#     count_m = []
#     count_r = []
#     i = 0
#     for question in train_questions:
#         print(i)
#         i +=1
#         mentions, relations = GetAmnswerSingle(question)
#         count_m.append(len(mentions))
#         count_r.append(len(relations[0]))

def overlap(question, relation):
    if len(relation) > 0:
        relation_ = ''.join(relation)
        n = 0
        for word in relation_:
            if word in question:
                n += 1
        if n == 0:
            return False
        else:
            return True
    else:
        return False


# import time
#
#
# question = '重庆位于中国的哪个地区？'
# start = time.time()
# mentions_ = ['重庆']
#
# relations = [GetRelationPaths(mentions_[0])]
# print(time.time() - start)
#
# pairs = []
# for i, mention in enumerate(mentions_):
#     for relation in relations[i]:
#         if relation:
#             if None not in relation:
#                 try:
#                     if overlap(question, relation):
#                         pairs.append([mentions_[i], relation])
#                 except:
#                     print(question, mentions_[i], relation)
# dataset = get_sim_example_features_eval(question, pairs, scoring.tokenizer, 50)
# start = time.time()
# ans = []
# for i in range(len(dataset)):
#     res = scoring.sim_model.predict(dataset[i])
#     ans.append(res)
# # res = scoring.batch_score(dataset)
# print(time.time() - start)
#
# for i in range(len(pairs)):
#     print(res[i*50])
#
# question_tensor = scoring.tokenizer.encode_plus(question, '重庆的总部', return_token_type_ids=True, return_tensors='tf')
#
#
# start = time.time()
# scores = []
# for item in pairs:
#     score = scoring.score(question, [item[0]], item[1])
#     scores.append([score, question, item[0], item[1]])
# print(time.time() - start)
#
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# tfidf = TfidfVectorizer()
# mentions_v = tfidf.fit_transform(mentions).toarray()
#
# from scipy import spatial
# from scipy.spatial.distance import cosine
# for i in range(2, len(mentions)):
#     cos_sim = 1 - cosine(mentions_v[1], mentions_v[i])
#     print(cos_sim, mentions[i])
#
# import numpy as np
# a = np.array([1,2])
# b = np.array([1,1.5])