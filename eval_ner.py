# -*- coding: utf-8 -*-
# @Time    : 2020/5/4 16:37
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : eval.py
# @Software: PyCharm



from transformers import (
    BertTokenizer,
    InputFeatures,
    BertConfig,
    TFBertForTokenClassification,
    TFBertForSequenceClassification,
    BertForSequenceClassification,
    BertForTokenClassification
)
import pickle
import numpy as np
import tensorflow as tf
# from data_process import ner_train_data
# from model import TFBertForTokenClassification
import os

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


vocab_file = 'F:\\july\\project\\roberta_chinese_wwm\\vocab.txt'
model_dir = 'save'

tokenizer = BertTokenizer.from_pretrained(vocab_file)
config = BertConfig.from_pretrained(model_dir)
model = TFBertForTokenClassification.from_pretrained(model_dir, config=config)




test_file = 'data/task1-4_valid_2020.questions'

with open(test_file, 'r', encoding='utf-8') as f:
    questions = f.readlines()
questions = [question.strip().split(':')[1] for question in questions]

def get_ner_entity(question):
    entities = ''
    question_token = tokenizer.tokenize(question)
    question_tensor = tokenizer.encode_plus(question, None, return_token_type_ids=True, return_tensors='tf' )
    prediction = model.predict(question_tensor)
    prediction = prediction.argmax(axis=-1)[0][1:-1]
    for i in range(len(prediction)):
        if prediction[i] == 0:
            entities += ' '
        else:
            entities += question_token[i]

    return entities.strip().split(' ')

entities = [get_ner_entity(question) for question in questions]
