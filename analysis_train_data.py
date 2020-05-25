# -*- coding: utf-8 -*-
# @Time    : 2020/5/14 23:06
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : analysis_train_data.py
# @Software: PyCharm


import pandas as pd
import pickle
train_csv_file = 'corpus/train_data.csv'
train_pkl_file = 'corpus/train_data.pkl'

train_data = pd.read_csv(train_csv_file, encoding='utf8')
train_data['count_m'] = train_data['gold_entities'].map(lambda x: len(x))



print(train_data[train_data['count_r'] == 10])





train_data = pickle.load(open(train_pkl_file, 'rb'))
train_questions = [train_data[i]['question'] for i in range(len(train_data))]
train_entities = [train_data[i]['gold_entities'] for i in range(len(train_data))]