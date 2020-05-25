# -*- coding: utf-8 -*-
# @Time    : 2020/4/29 22:34
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : make_kb_file.py
# @Software: PyCharm


import pandas as pd
import csv
import pickle


def make_neo4j_file():
    #读取三元组
    h_r_t_name = [":START_ID", 'name', ":END_ID"]
    # kb_file = 'data/pkubase-complete.txt'
    # h_r_t = []
    # with open(kb_file, 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         line = line[:-2].strip().replace('"', '').replace('<', '').replace('>', '')
    #         h_r_t.append(line)
    #
    # with open('data/pkubase-complete_.txt', 'w', encoding='utf-8') as f:
    #     for line in h_r_t:
    #         f.write(line + '\n')
    # # 66499746

    kb_file = 'data/pkubase-complete_.txt'
    # h_r_t = pd.read_csv(kb_file, names=h_r_t_name)
    h_r_t = pd.read_table(kb_file, decimal='\t', names=h_r_t_name, encoding='utf-8')
    # h_r_t = h_r_t[:10000]

    # len 66498316
    # 66458370
    print(h_r_t.info())
    print(h_r_t.head())


    # 去除重复实体
    # h_r_t[':START_ID'] = h_r_t[':START_ID'].map(lambda x: str(x).lower())
    # h_r_t[':END_ID'] = h_r_t[':END_ID'].map(lambda x: str(x).lower())
    entity = set()
    entity1 = set()  # 用于分词筛选实体
    entity_h = h_r_t[':START_ID'].tolist()
    entity_t = h_r_t[':END_ID'].tolist()
    for i in entity_h:
        entity.add(i)
        entity1.add(i)

    for i in entity_t:
        entity.add(i)
    print(len(entity))
    print(len(entity1))
    # 25573820
    # 25573727
    # 25573764
    # 17609390
    # 17556701

    # 保存节点文件
    csvf_entity = open('data/entity.csv', 'w', newline='', encoding='utf-8')
    w_entity = csv.writer(csvf_entity)
    # 实体ID，要求唯一，名词，LABEL标签，可自己不同设定对应的标签
    w_entity.writerow(('entity:ID', 'name', ':LABEL'))
    entity = list(entity)
    entity_dict = {}
    for i in range(len(entity)):
        w_entity.writerow(('e' + str(i), entity[i], 'Entity'))
        entity_dict[entity[i]] = 'e' + str(i)
    csvf_entity.close()
    pickle.dump(entity_dict, open('corpus/entity_dic.pkl', 'wb'))

    with open('corpus/entity_dic.txt', 'w', encoding='utf-8') as f:
        for i in entity1:
            f.write(str(i) + '\n')

    # 生成relation 词典，用于分词时筛选无意义的entity
    entity_t = h_r_t['name'].tolist()
    relations = set()
    for i in entity_t:
        relations.add(i)
    print(len(relations))
    with open('corpus/relation_dic.txt', 'w', encoding='utf-8') as f:
        for i in relations:
            f.write(str(i) + '\n')

    # 生成关系文件，起始实体ID， 终点实体ID， 要求与实体文件中ID对应， ：TYPE为'Relation', name为relation name


    h_r_t[':START_ID'] = h_r_t[':START_ID'].map(entity_dict)
    h_r_t[':END_ID'] = h_r_t[':END_ID'].map(entity_dict)
    h_r_t[':TYPE'] = 'Relation'
    h_r_t = h_r_t[[':START_ID', ':END_ID', ':TYPE', 'name']]

    h_r_t.to_csv('data/roles.csv', index=False, encoding='utf-8')

# 导入neo4j的命令
# .\neo4j-admin.bat import --nodes:Entity F:\\july\\project\\game\\CCKS2020-CKBQA\\data\\entity.csv --relationships F:\\july\\project\\game\\CCKS2020-CKBQA\\data\\roles.csv


def make_mention_dic():
    mention_file = 'data/pkubase-mention2ent.txt'
    mention_vocab = set()
    mention_dic = {}
    with open(mention_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            mentions = line.strip().split('\t')
            mention_vocab.add(mentions[0])
            mention_vocab.add(mentions[1])
            if mentions[0].lower() not in mention_dic:
                mention_dic[mentions[0].lower()] = [mentions[0]]
                mention_dic[mentions[0].lower()] = mention_dic[mentions[0].lower()] + [mentions[1]]
            else:
                mention_dic[mentions[0].lower()] = mention_dic[mentions[0].lower()] + [mentions[1]]

    with open('corpus/mention_vocab.txt', 'w', encoding='utf8') as f:
        for i in mention_vocab:
            f.write(str(i) + '\n')

    pickle.dump(mention_dic, open('corpus/mention_dic.pkl', 'wb'))

    # 将词典加入到jieba默认词典
    # with open('C:\\Users\\daish\\anaconda3\\Lib\\site-packages\\jieba\\dict.txt', 'a', encoding='utf8') as f:
    #     for i in mention_vocab1:
    #         f.write(str(i).replace(' ', '') + ' 3' + ' n\n')
    #     for i in mention_vocab2:
    #         f.write(str(i).replace(' ', '') + ' 99' + ' n\n')


if __name__ == '__main__':
    make_mention_dic()
