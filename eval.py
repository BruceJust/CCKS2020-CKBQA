# -*- coding: utf-8 -*-
# @Time    : 2020/5/17 18:40
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : eval.py
# @Software: PyCharm

from answering_new import GetAmnswer
import datetime

Ans = GetAmnswer()

def eval(input_path, out_path1, out_path2):
    ans_list = []
    tuple_list = []
    with open(input_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f.readlines()):
            question = line.strip().split(':')[1]
            ans = Ans.getanswer(question)
            ans_list.append(ans)
            tuple_list.append((line, ans))
            print(question, ans , '\n')
            print(i, datetime.datetime.now())

    with open('corpus/mention_vocab.txt', 'r', encoding='utf-8') as f:
        mention_vocab = {}
        for line in f:
            if line.strip():
                mention_vocab[line.strip()] = 0

    ans_list_new = []
    for item in ans_list:
        if item != '':
            items = item.split('\t')
            new_items = []
            entity_flag = True
            for i in items:
                if i not in Ans.extractor.mention_dic:
                    entity_flag = False
            if entity_flag == True:
                for i in items:
                    i = '<' + str(i) + '>'
                    new_items.append(i)
            else:
                for i in items:
                    i = '"' + str(i) + '"'
                    new_items.append(i)
            new_items = '\t'.join(new_items)
        else:
            new_items = item
        ans_list_new.append(new_items)


    with open(out_path1, 'w', encoding='utf8') as f:
        for line in ans_list_new:
            f.write(str(line) + '\n')

    with open(out_path2, 'w', encoding='utf8') as f:
        for line in tuple_list:
            f.write(str(line[0]) + ' ||| ' +  str(line[1])  + '\n')


if __name__ == '__main__':
    input_path = 'data/task1-4_valid_2020.questions'
    out_path1 = 'output/result.txt'
    out_path2 = 'output/result_pair.txt'
    eval(input_path, out_path1, out_path2)



with open('output/result_20200524.txt', 'r', encoding='utf8') as f:
    ans_list_new_ = []
    for i, item in enumerate(f.readlines()):
        item = item.strip()
        if item != '':
            items = item.split('\t')
            new_items = []
            for i in items:
                if i in mention_vocab:
                    i = '<' + str(i) + '>'
                else:
                    i = '"' + str(i) + '"'
                new_items.append(i)
            new_items = '\t'.join(new_items)
        else:
            new_items = item
        ans_list_new_.append(new_items)