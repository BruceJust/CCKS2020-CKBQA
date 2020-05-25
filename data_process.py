# -*- coding: utf-8 -*-
# @Time    : 2020/4/26 21:23
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : data_process.py
# @Software: PyCharm

import re
import pandas as pd
import pickle
import tensorflow as tf
from transformers import InputFeatures
import random

def get_ner_file(inputpath, outputpath):
    question_num = 0
    e1hop1_num = 0
    e1hop2_num = 0
    e2hop2_num = 0
    with open(inputpath, 'r', encoding='utf-8') as f:
       a = f.readlines()
    n = len(a) // 4 + 1
    train_data = []
    train_data_ = {}
    for i in range(n):
        example = a[i * 4: (i +1 ) * 4]
        question = example[0].strip()
        qid = question.split(':')[0]
        question = question.split(':')[1]
        sql = re.findall('{.+}', example[1])[0]
        elements = re.findall('<.+?>|\".+?\"|\?\D', sql) + re.findall('\".+?\"', sql)
        new_elements = []
        for e in elements:
            if e[0] == '\"':
                if e not in new_elements:
                    new_elements.append(e)
            else:
                new_elements.append(e)
        elements = new_elements
        gold_entities = []
        gold_relations = []
        for j in range(len(elements)):
            if elements[j][0] == '<' or elements[j][0] == '\"':
                if j % 3 == 1:
                    gold_relations.append(elements[j])
                else:
                    gold_entities.append(elements[j])

        answer = example[2].strip().replace('<', '').replace('>', '')
        dic = {}
        dic['question'] = question
        dic['answer'] = answer#问题的答案
        dic['gold_entities'] = gold_entities
        dic['gold_relations'] = gold_relations
        dic['sql'] = sql
        train_data_[i] = dic

        train_data.append((qid, question, answer, gold_entities, gold_relations, sql, len(gold_entities), len(gold_relations)))
        if len(gold_entities) == 1 and len(gold_relations) == 1:
            e1hop1_num += 1
        elif len(gold_entities) == 1 and len(gold_relations) == 2:
            e1hop2_num += 1
        elif len(gold_entities) == 2 and len(gold_relations) == 2:
            e2hop2_num += 1
        elif len(gold_entities) == 2 and len(gold_relations) < 2:
            print(elements)
            print(gold_entities)
            print(sql)
            print('\n')
    question_num += 1
    print ('语料集问题数为%d==单实体单关系数为%d====单实体双关系数为%d==双实体双关系数为%d==总比例为%.3f\n'\
           %(question_num,e1hop1_num,e1hop2_num,e2hop2_num,(e1hop1_num+e1hop2_num+e2hop2_num)/question_num))
    train_data = pd.DataFrame(train_data, columns=['qid', 'question', 'answer', 'gold_entities', 'gold_relations', 'sql', 'count_m', 'count_r'])
    train_data.to_csv(outputpath + '.csv', encoding='utf-8', index=False)
    pickle.dump(train_data_, open(outputpath + '.pkl', 'wb'))

def ner_train_data(file1, file2, tokenizer, max_seq_len):
    train_data = pickle.load(open(file1, 'rb'))
    if file2:
        train_data2 =  pickle.load(open(file2, 'rb'))
        for i in range(len(train_data2)):
            train_data[len(train_data)] = train_data2[i]
    train_questions = [train_data[i]['question'] for i in range(len(train_data))]
    train_entities = [train_data[i]['gold_entities'] for i in range(len(train_data))]
    train_entities = [[entity[1:-1].split('_')[0] for entity in line] for line in train_entities] # 把长的entity缩短，并去除<>
    return get_ner_example_features(train_questions, train_entities, tokenizer, max_seq_len), len(train_questions)

def sim_train_data(file1, file2, tokenizer, max_seq_len):
    train_data = pickle.load(open(file1, 'rb'))
    if file2:
        train_data2 =  pickle.load(open(file2, 'rb'))
        for i in range(len(train_data2)):
            train_data[len(train_data)] = train_data2[i]
    train_questions = [train_data[i]['question'] for i in range(len(train_data))]
    train_entities = [train_data[i]['gold_entities'] for i in range(len(train_data))]
    train_relations = [train_data[i]['gold_relations'] for i in range(len(train_data))]
    train_entities = [[entity[1:-1].split('_')[0] for entity in line] for line in train_entities] # 把长的entity缩短，并去除<>
    train_relations = [[relation[1:-1] for relation in line] for line in train_relations] #去除<>
    return get_sim_example_features(train_questions, train_entities, train_relations, tokenizer, max_seq_len), len(train_questions)

def find_lcsubstr(s1, s2):
    m=[[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0  #最长匹配的长度
    p=0 #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
            if m[i+1][j+1]>mmax:
                mmax=m[i+1][j+1]
                p=i+1
    return s1[p-mmax:p]

def get_ner_example_features(questions, entities, tokenizer, max_seq_len):
    features = []
    for i in range(len(questions)):
        q = questions[i]
        inputs = tokenizer.encode_plus(
            q, None, add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=True,
            pad_to_max_length=True
        )
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        y = [[0] for j in range(max_seq_len)]
        assert len(input_ids)==len(y)
        for e in entities[i]:
            #得到实体名和问题的最长连续公共子串
            e = find_lcsubstr(e,q)
            if e in q:
                begin = q.index(e)+1
                end = begin + len(e)
                if end < max_seq_len-1:
                    for pos in range(begin,end):
                        y[pos] = [1]
        y = [j[0] for j in y]
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=y
            )
        )

    def gen():
        for ex in features:
            yield (
                {
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "token_type_ids": ex.token_type_ids,
                },
                ex.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
         tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([None]),
        )

    )


def get_sim_example_features(questions, entities, relations, tokenizer, max_seq_len):
    features = []
    n = 3
    for i in range(len(questions)):
        q = questions[i]
        entity = '的'.join(entities[i])
        relation = '的'.join(relations[i])
        # # x = [[0] for j in range(len(q))]
        # for e in entities[i]:
        #     #得到实体名和问题的最长连续公共子串
        #     e = find_lcsubstr(e,q)
        #     q = q.replace(e, '<sub>')
        predicate = entity + '的' + relation
        y = 1
        inputs = tokenizer.encode_plus(
            q, predicate, add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=True,
            pad_to_max_length=True
        )
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
            "attention_mask"]

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=y
            )
        )
        # add negative sampling
        for j in range(n):
            y = 0
            idx = random.randint(0, len(questions) - 1)
            if idx != i:
                relation = '的'.join(relations[idx])
                predicate = entity + '的' + relation
                inputs = tokenizer.encode_plus(
                    q, predicate, add_special_tokens=True,
                    max_length=max_seq_len,
                    return_token_type_ids=True,
                    pad_to_max_length=True
                )
                input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
                    "attention_mask"]

                features.append(
                    InputFeatures(
                        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=y
                    )
                )

    def gen():
        for ex in features:
            yield (
                {
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "token_type_ids": ex.token_type_ids,
                },
                ex.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
         tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        )

    )


def get_sim_example_features_eval(question, items, tokenizer, max_seq_len, back=False):
    features = []
    q = question
    for i in range(len(items)):
        entity = '的'.join(items[i][1])
        relation = '的'.join(items[i][2])
        # # x = [[0] for j in range(len(q))]
        # for e in entities[i]:
        #     #得到实体名和问题的最长连续公共子串
        #     e = find_lcsubstr(e,q)
        #     q = q.replace(e, '<sub>')
        if back:
            predicate = relation + '的' + entity
        else:
            predicate = entity + '的' + relation
        inputs = tokenizer.encode_plus(
            q, predicate,
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors='tf'
        )
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
            "attention_mask"]

        features.append(
            inputs
            )

    return features



def make_tf_datasets(features):
    def gen():
        for ex in features:
            yield (
                {
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "token_type_ids": ex.token_type_ids,
                },
                ex.label,
            )
    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32 },
         tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([None]),
        )

    )


def process_nlpcc2016_corpus():
    train_file = 'data/nlpcc-iccpol-2016.kbqa.training-data'
    test_file = 'data/nlpcc-iccpol-2016.kbqa.testing-data'
    f_train = open(train_file, 'r', encoding='utf8')
    f_test = open(test_file, 'r', encoding='utf8')

    train_data_ = {}
    i = 0
    q = ''
    for line in f_train:
        if line.find('<q') == 0:  #question line
            q = line[line.index('>') + 2:].strip()
            continue
        elif line.find('<t') == 0:  #triple line
            triple = line[line.index('>') + 2:]
            e = triple[:triple.index(' |||')].strip()
            triNS = triple[triple.index(' |||') + 5:]
            r = triNS[:triNS.index(' |||')]

            dic = {}
            dic['question'] = q  # 原问题
            dic['answer'] = ' '
            dic['gold_entities'] = ['<'+ str(e) + '>']  # 实体
            dic['gold_relations'] = ['<'+ str(r) + '>']  # relation
            dic['sql'] = line
            train_data_[i] = dic
            i += 1
        else:
            continue

    for line in f_test:
        if line.find('<q') == 0:  # question line
            q = line[line.index('>') + 2:].strip()
            continue
        elif line.find('<t') == 0:  # triple line
            triple = line[line.index('>') + 2:]
            e = triple[:triple.index(' |||')].strip()
            triNS = triple[triple.index(' |||') + 5:]
            r = triNS[:triNS.index(' |||')]

            dic = {}
            dic['question'] = q  # 原问题
            dic['answer'] = ' '
            dic['gold_entities'] = ['<'+ str(e) + '>']  # 实体
            dic['gold_relations'] = ['<'+ str(r) + '>']  # relation
            dic['sql'] = line
            train_data_[i] = dic
            i += 1
        else:
            continue
    pickle.dump(train_data_, open('corpus/train_data2016.pkl', 'wb'))

if __name__ == '__main__':
    train_file = 'data\\task1-4_train_2020.txt'
    valid_file = 'data\\task6ckbqa_valid_2019.txt'
    train_outpath = 'corpus\\train_data'
    valid_outpath = 'corpus\\valid_data'
    get_ner_file(train_file, train_outpath)
    get_ner_file(valid_file, valid_outpath)