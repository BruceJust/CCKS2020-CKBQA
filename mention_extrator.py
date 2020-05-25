# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 21:50
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : mention_extrator.py
# @Software: PyCharm

import re
import jieba
import time
import pickle
import numpy as np
import tensorflow as tf
from transformers import (TFBertForTokenClassification,BertTokenizer,
    BertConfig,
    TFBertForSequenceClassification,
    TFBertForTokenClassification)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model_ner_dir = 'model_ner'
model_sim_dir = 'model_sim'
model_sim_dir_ = 'model_sim_new'

class MentionExtractor(object):
    def __init__(self):
        with open('corpus/relation_dic.txt', 'r', encoding='utf-8') as f:
            relation_dic = {}
            for line in f:
                if line.strip():
                    relation_dic[line.strip()] = 0
        self.relation_dic = relation_dic
        mention_dic = pickle.load(open('corpus/mention_dic.pkl', 'rb'))
        self.mention_dic = mention_dic
        entity_dic = pickle.load(open('corpus/entity_dic.pkl', 'rb'))
        self.entity_dic = entity_dic
        self.max_len = 50
        start = time.time()
        jieba.lcut('我的测试')
        # jieba.load_userdict('corpus/mention_vocab.txt')
        print('loaded entity dict for: %.2f'%(time.time() - start))
        self.tokenizer = BertTokenizer.from_pretrained(model_ner_dir)
        self.ner_model = TFBertForTokenClassification.from_pretrained(model_ner_dir)
        print('mention extractor loaded')

    def extract_mentions(self, question):


        # jieba获取一部分候选 mention

        mentions = jieba.lcut(question)

        # 使用ner获取一部分候选 mention, 暂时保持相同的权重，后期考虑BERT的给与更高的权重
        length, mentions_bert = self.get_ner_entity(question)

        if length > 1:  # 如果是对实体则特殊处理
            mentions_bert_ = []
            for i in mentions_bert:
                for j in i:
                    mentions_bert_.append(j['name'])
        elif length == 1:
            mentions_bert_ = []
            for i in mentions_bert:
                mentions_bert_.append(i['name'])
        else:
            mentions_bert_ = mentions_bert

        mentions = mentions + mentions_bert_

        # 去除太短的或者无意义的mention或明显是relation的
        stop_mention_dic = {'', '的', '我的', '是什么', '什么', '哪些', '哪个', '哪里' , '出生', '董事长',
                            '生的', '多少', '名下', '企业' ,'作者', '故乡' ,'在哪' ,'主席' , '是谁',
                            '现任', '董事长', '儿子', '演唱者' ,'类型', '何时' , '生于', '何时'}
        mentions_list = list(set(mentions))
        mentions = []
        for mention in mentions_list:
            if len(mention) > 1 and mention not in stop_mention_dic:
                mentions.append(mention)
        # 将所有为relation的entity去掉
        # mentions_list = list(set(mentions))
        # mentions = []
        # for mention in mentions_list:
        #     if mention not in self.relation_dic:
        #         mentions.append(mention)

        # 加入重名候选mention的细分
        mentions_ = []
        for mention in mentions:
            mention_ = self.mention_dic.get(mention, 0)
            if mention_ != 0:
                for item in mention_:
                    mentions_.append(item)
        mentions = mentions + mentions_

        # 仅保留在kb有的entity
        mentions_list = mentions
        mentions = []
        for token in mentions_list:
            if token in self.entity_dic:
                mentions.append(token)


        # 去除重复
        mentions = list(set(mentions))

        # 获取entity 号
        entities_result= {}
        for i in range(len(mentions)):
            entity_d = {}
            entity_d['name'] = mentions[i]
            entity_d['entity'] = self.entity_dic[mentions[i]]
            entities_result[i] = entity_d

        return entities_result

    def get_ner_entity(self, question):
        entities = ''
        question_token = self.tokenizer.tokenize(question)
        question_tensor = self.tokenizer.encode_plus(question, None, return_token_type_ids=True, return_tensors='tf')
        prediction = self.ner_model.predict(question_tensor)
        prediction = prediction.argmax(axis=-1)[0][1:-1] # 取出每个字的得分并去除[CLS] [SEP]
        # for i in range(len(prediction)):
        #     if prediction[i] == 0:
        #         entities += ' '
        #     else:
        #         entities += question_token[i]
        # entities = entities.strip().split(' ')
        for i in range(len(prediction)):
            if prediction[i] == 0:
                question_token[i] = ' '
        entities = self.convert_tokens_to_string(question_token)
        entities = set(entities)
        entities_ = entities
        entities = []
        for entity in entities_:
            if len(entity) > 1 :
                entities.append(entity)
        if len(entities) == 0:
            return 0, entities
        elif len(entities) == 1:
            if entities[0] not in self.entity_dic:
                return 0, []
            else:
                entities_ = []
                if self.entity_dic.get(entities[0], 0) != 0:
                    entities_ = [{'name': entities[0], 'entity': self.entity_dic[entities[0]]}]
                    mentions = self.mention_dic.get(entities[0], 0)
                    if mentions != 0:
                        for j in mentions:
                            if self.entity_dic.get(j, 0) != 0 and j != entities[0]:
                                entities_.append({'name': j, 'entity': self.entity_dic[j]})
                    return 1, entities_
                else:
                    return 0, entities_
        else:
            for item in entities:
                if item not in self.entity_dic: # 如果有一个entity无法链接kb，那么仍然只能按单entity计算
                    entities.remove(item)
            if len(entities) == 0:
                return 0, entities
            elif len(entities) == 1:
                if entities[0] not in self.entity_dic:
                    return 0, []
                else:
                    entities_ = []
                    if self.entity_dic.get(entities[0], 0) != 0:
                        entities_ = [{'name': entities[0], 'entity': self.entity_dic[entities[0]]}]
                        mentions = self.mention_dic.get(entities[0], 0)
                        if mentions != 0:
                            for j in mentions:
                                if self.entity_dic.get(j, 0) != 0 and j != entities[0]:
                                    entities_.append({'name': j, 'entity': self.entity_dic[j]})
                        return 1, entities_
                    else:
                        return 0, entities_
            else:
                entities_dic = []
                for i in range(len(entities)):
                    if self.entity_dic.get(entities[i], 0) != 0:
                        entities_dic.append([{'name': entities[i], 'entity': self.entity_dic[entities[i]]}])
                    else:
                        entities_dic.append([])
                    mentions = self.mention_dic.get(entities[0], 0)
                    if mentions != 0:  # 将所有可能双实体列出
                        for j in mentions:
                            if self.entity_dic.get(j, 0) != 0 and j != entities[i]:
                                entities_dic[i].append({'name':j, 'entity':self.entity_dic[j]})
                return len(entities_dic), entities_dic


    def GetEntityMention(self, corpus):
        mention_num = 0
        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            dic['entity_mention'] = self.extract_mentions(question)
            corpus[i] = dic
            # print(question)
            # print(dic['entity_mention'])
            mention_num += len(dic['entity_mention'])
        # print(mention_num, len(corpus))
        return corpus

    def convert_tokens_to_string(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        context = ''.join(tokens)
        match =  re.findall('[^\x00-\xff]', context)
        if len(match) > 0:  # 按中文处理
            text = text.split('   ')
            for i in range(len(text)):
                text[i] = text[i].replace(' ', '')
        else:
            text = [text]
        return text




class Bert_scoring(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(model_sim_dir)
        self.sim_model = TFBertForSequenceClassification.from_pretrained(model_sim_dir)
        self.sim_model_ = TFBertForSequenceClassification.from_pretrained(model_sim_dir_)

    def score(self,question, entity, relation):
        predicate = '的'.join(entity) + '的' + '的'.join(relation)
        question_tensor = self.tokenizer.encode_plus(question, predicate, return_token_type_ids=True, return_tensors='tf')
        prediction = self.sim_model.predict(question_tensor)
        return prediction[0][1]

    def score_(self,question, entity, relation):
        predicate = '的'.join(entity) + '的' + '的'.join(relation)
        question_tensor = self.tokenizer.encode_plus(question, predicate, return_token_type_ids=True, return_tensors='tf')
        prediction = self.sim_model_.predict(question_tensor)
        return prediction[0][1]

    def batch_score(self, data):
        prediction = self.sim_model.predict(data)
        return prediction


# import re
# a = re.findall('[^\x00-\xff]','京东方a')