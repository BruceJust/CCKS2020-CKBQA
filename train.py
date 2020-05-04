# -*- coding: utf-8 -*-
# @Time    : 2020/5/3 17:27
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : train.py
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
from data_process import ner_train_data
# from model import TFBertForTokenClassification
import os

vocab_file = 'F:\\july\\project\\roberta_chinese_wwm\\vocab.txt'
config_file = 'F:\\july\\project\\roberta_chinese_wwm\\config.json'
bert_file = 'F:\\july\\project\\roberta_chinese_wwm'
num_labels = 2
max_seq_len = 25
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
tokenizer = BertTokenizer.from_pretrained(vocab_file)
config = BertConfig.from_pretrained(config_file, num_labels=num_labels)
# model = BertForTokenClassification.from_pretrained(bert_file, config=config, from_)
# model.save_pretrained('roberta')
model = TFBertForTokenClassification.from_pretrained(bert_file, config=config, from_pt=True)


train_file = 'corpus\\train_data.pkl'
valid_file = 'corpus\\valid_data.pkl'


train_data, train_num = ner_train_data(train_file, tokenizer, max_seq_len)
valid_data, valid_num = ner_train_data(valid_file, tokenizer, max_seq_len)
train_data = train_data.shuffle(128).batch(BATCH_SIZE).repeat(-1)
valid_data = valid_data.batch(EVAL_BATCH_SIZE)


opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=opt, loss=loss, metrics=[metric])

train_steps = train_num // BATCH_SIZE
valid_steps = valid_num // EVAL_BATCH_SIZE
model_dir = 'save'
os.makedirs(model_dir, exist_ok=True)
model_file = os.path.join(model_dir, 'tf_model.h5')
callbacks = [tf.keras.callbacks.ModelCheckpoint(model_file,
                                                monitor='val_accuracy',
                                                save_best_only=True)
             ]
model.save_pretrained(model_dir)
history = model.fit(
    train_data,
    epochs=5,
    steps_per_epoch=train_steps,
    validation_data=valid_data,
    validation_steps=valid_steps,
    verbose=1,
    callbacks=callbacks,
)
