# -*- coding: utf-8 -*-
# @Time    : 2020/4/29 22:34
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : make_kb_file.py
# @Software: PyCharm


import pandas as pd
import csv

#读取三元组
h_r_t_name = [":START_ID", 'role', ":END_ID"]
kb_file = 'data/pkubase-complete.txt'
h_r_t = []
with open(kb_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line[:-2].strip().replace('"', '').replace('<', '').replace('>', '')
        h_r_t.append(line)

with open('data/pkubase-complete_.txt', 'w', encoding='utf-8') as f:
    for line in h_r_t:
        f.write(line + '\n')
# 66499746




kb_file = 'data/pkubase-complete_.txt'
# h_r_t = pd.read_csv(kb_file, names=h_r_t_name)
h_r_t = pd.read_table(kb_file, decimal='\t', names=h_r_t_name, encoding='utf-8')

# len 66498316
# 66458370
print(h_r_t.info())
print(h_r_t.head())




# 去除重复实体
entity = set()
entity_h = h_r_t[':START_ID'].tolist()
entity_t = h_r_t[':END_ID'].tolist()
for i in entity_h:
    entity.add(i)
    entity.add(str(i).split('_')[0])

for i in entity_t:
    entity.add(i)
    entity.add(str(i).split('_')[0])
print(len(entity))
# 25573820
# 25573727
# 25573764

# 保存节点文件
csvf_entity = open('data/entity.csv', 'w', newline='', encoding='utf-8')
w_entity = csv.writer(csvf_entity)
# 实体ID，要求唯一，名词，LABEL标签，可自己不同设定对应的标签
w_entity.writerow(('entity:ID', 'name', ':LABEL'))
entity = list(entity)
entity_dict = {}
for i in range(len(entity)):
    w_entity.writerow(('e' + str(i), entity[i], 'my_entity'))
    entity_dict[entity[i]] = 'e' + str(i)
csvf_entity.close()

# 生成关系文件，起始实体ID， 重点实体ID， 要求与实体文件中ID对应， ：TYPE即为关系
h_r_t[':START_ID'] = h_r_t[':START_ID'].map(entity_dict)
h_r_t[':END_ID'] = h_r_t[':END_ID'].map(entity_dict)
h_r_t[':TYPE'] = h_r_t['role']
h_r_t.pop('role')
h_r_t.to_csv('data/roles.csv', index=False, encoding='utf-8')

# 导入neo4j的命令
# .\neo4j-admin.bat import --nodes F:\\july\\project\\game\\CCKS2020-CKBQA\\data\\entity.csv --relationships F:\\july\\project\\game\\CCKS2020-CKBQA\\data\\roles.csv


h_r_t = h_r_t[~((h_r_t[':START_ID']=='e11430533')&(h_r_t[':END_ID']=='e7569145'))]
h_r_t = h_r_t[~(h_r_t[':END_ID']==None)]
a = h_r_t[':TYPE'].drop_duplicates()