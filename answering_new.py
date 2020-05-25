# -*- coding: utf-8 -*-
# @Time    : 2020/5/10 21:58
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : main.py
# @Software: PyCharm

from mention_extrator import MentionExtractor, Bert_scoring
from kb import  (
    GetRelationPathsSingle,
    GetEntity2TwoRelation,
    GetTwoEntityRelation,
    GetEntity2Tuple,
    GetRelationPathsBack,
    GetRelationPathsForward,
    GetAnsFword,
    GetAnsBack
)
import pickle
import time
from data_process import get_sim_example_features_eval

# train_file = 'corpus\\train_data.pkl'
# train_data = pickle.load(open(train_file, 'rb'))
# train_questions = [train_data[i]['question'] for i in range(len(train_data))]



class GetAmnswer:
    def __init__(self):
        self.extractor = MentionExtractor()
        self.scoring = Bert_scoring()

    def getanswer(self, question):
        start1 = time.time()
        length, mentions_ner = self.extractor.get_ner_entity(question)
        mentions_ = self.extractor.extract_mentions(question)
        print(time.time() - start1)
        ans = ''
        if length == 0:
            # 如果无ner则走分词结果，但只关注正向relation
            ans = self.process_one_entity_forward(question, mentions_)
        elif length == 1:
            # 如果ner为1，则以ner优先，正向relation无结果，再看反向relation，如果ner无结果，再走分词正向relation
            ans = self.process_one_entity(question, mentions_ner)
            if ans == '':
                ans = self.process_one_entity_forward(question, mentions_)
        elif length == 2:
            # ner为双实体，则先走双实体双向关系，如果无答案，再走单实体ner，无结果，再走分词正向relation
            ans = self.process_two_entity(question, mentions_ner)
            if ans == '':
                mentions_ner_ = []
                for i in mentions_ner:
                    for j in i:
                        mentions_ner_.append(j)
                ans = self.process_one_entity(question, mentions_ner_)
                if ans == '':
                    ans = self.process_one_entity_forward(question, mentions_)
        else:
            # ner 超过2个，暂时按单实体处理
            mentions_ner_ = []
            for i in mentions_ner:
                for j in i:
                    mentions_ner_.append(j)
            ans = self.process_one_entity(question, mentions_ner_)
            if ans == '':
                ans = self.process_one_entity_forward(question, mentions_)
        print('finish with', time.time() - start1)
        return ans


    def process_one_entity(self, question, mentions):
        ans = self.process_one_entity_forward(question, mentions)
        if ans == '':
            ans = self.process_one_entity_back(question, mentions)
        return ans

    def process_one_entity_forward(self, question, mentions):
        start = time.time()
        relations = [GetRelationPathsForward(mentions[i]) for i in range(len(mentions))]
        print('searching relations for', time.time() - start)

        pairs = []
        for i in range(len(mentions)):
            for relation in relations[i]:
                if relation:
                    if None not in relation:  # 个别 relation name是 None
                        if overlap(question, [mentions[i]['name']], relation):
                            pairs.append([[mentions[i]], [mentions[i]['name']], relation])

        # 用BERT计算相似度
        scores = self.score_fun(question, pairs, back=False)
        max_score = 0
        best_pattern = []
        for item in scores:
            if item[0] > max_score:
                max_score = item[0]
        for item in scores:
            if item[0] == max_score:
                best_pattern.append(item)
        if len(best_pattern) == 0:
            ans = ''
        elif len(best_pattern) == 1:
            ans = GetAnsFword(best_pattern[0][2][0], best_pattern[0][3])
        # 如果有多个候选relation一样的分数，优先选relation少的
        else:
            ans = ''
            min_len = 3
            for item in best_pattern:
                if len(item[3]) < min_len:
                    min_len = len(item[3])
            for item in best_pattern:
                if len(item[3]) == min_len:
                    ans = GetAnsFword(item[2][0], item[3][0])
        return ans

    def process_one_entity_back(self, question, mentions):
        start = time.time()
        relations = [GetRelationPathsBack(mentions[i]) for i in range(len(mentions))]
        print('searching relations for', time.time() - start)

        pairs = []
        for i in range(len(mentions)):
            for relation in relations[i]:
                if relation:
                    if None not in relation:  # 个别 relation name是 None
                        if overlap(question, [mentions[i]['name']], relation):
                            pairs.append([[mentions[i]], [mentions[i]['name']], relation])

        # 用BERT计算相似度
        scores = self.score_fun(question, pairs, back=True)
        max_score = 0
        best_pattern = []
        for item in scores:
            if item[0] > max_score:
                max_score = item[0]
        for item in scores:
            if item[0] == max_score:
                best_pattern.append(item)
        if len(best_pattern) == 0:
            ans = ''
        elif len(best_pattern) == 1:
            ans = GetAnsBack(best_pattern[0][2][0], best_pattern[0][3])
        # 如果有多个候选relation一样的分数，优先选relation少的
        else:
            ans = ''
            # min_len = 2
            # for item in best_pattern:
            #     if len(item[3]) < min_len:
            #         min_len = len(item[3])
            # for item in best_pattern:
            #     if len(item[3]) == min_len:
            #         ans = GetAnsBack(item[2][0], item[3][0])
        return ans


    def process_two_entity(self, question, mentions):
        mention_pairs = []
        for i in mentions[0]:
            for j in mentions[1]:
                mention_pairs.append([i, j])
        relations = [GetTwoEntityRelation(mention[0], mention[1]) for mention in mention_pairs]
        if len(relations) == 1 and len(relations[0]) == 0:
            # 若无relation，直接返回
            return ''
        pairs = []
        for i in range(len(mention_pairs)):
            for relation in relations[i]:
                if relation:
                    if None not in relation:  # 个别 relation name是 None
                        # if overlap(question, [mention['name'] for mention in mention_pairs[i]], relation):
                            pairs.append([[mention_pairs[i]], [mention_pairs[i][0]['name'], mention_pairs[i][1]['name']], relation])

        if len(pairs) == 0:
            # 若无实质relation，直接返回
            return ''

        # 用BERT计算相似度
        scores_two = self.score_fun(question, pairs, back=False)
        max_score = 0
        best_pattern = []
        for item in scores_two:
            if item[0] > max_score:
                max_score = item[0]
        for item in scores_two:
            if item[0] == max_score:
                best_pattern.append(item)
        if len(best_pattern) == 0:
            ans = ''
        else:
            ans = GetEntity2Tuple(best_pattern[0][2][0][0], best_pattern[0][3][0], best_pattern[0][3][1],
                                  best_pattern[0][2][0][1])
        return ans

    def score_fun(self, question, pairs, back=False):
        dataset = get_sim_example_features_eval(question, pairs, self.scoring.tokenizer, 50, back)
        start = time.time()
        # 相似度取两个model平均
        # res1 = []
        # for i in range(len(dataset)):
        #     res = self.scoring.sim_model.predict(dataset[i])
        #     res1.append(res[0][1])
        res2 = []
        for i in range(len(dataset)):
            res = self.scoring.sim_model_.predict(dataset[i])
            res2.append(res[0][1])
        scores = []
        for i in range(len(dataset)):
            scores.append([res2[i], question, pairs[i][0], pairs[i][2]])
        print(f'scoring for {len(dataset)} relations with', time.time() - start)
        return scores

def overlap(question, mention, relation):
    if len(relation) > 0:
        relation_ = ''.join(relation)
        mention_ = ''.join(mention)
        text = relation_ + mention_
        n = 0
        for word in text:
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
#
# dataset = get_sim_example_features_eval(question, pairs, scoring.tokenizer, 50)
# start = time.time()
# res = scoring.sim_model.predict_generator(dataset)
# res = scoring.sim_model.predict_on_batch(dataset.take(1))
# # res = scoring.batch_score(dataset)
# print(f'scoring for {len(dataset)} relations with', time.time() - start)