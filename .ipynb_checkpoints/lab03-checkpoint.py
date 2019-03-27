# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:20:47 2018

@author: MaoChuLin
"""
import os
import re
import io
import json
import datetime

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import jieba

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import keras
from keras.models import Model, Sequential, load_model, model_from_json
from keras.layers import Input, LSTM, Dense, Embedding, GRU, Bidirectional
from keras.utils import to_categorical

en_tokenize = lambda data: [word_tokenize(sen) for sen in data]
ch_tokenize = lambda data: [list(jieba.cut(sen)) for sen in data]

def load_file(filename):
    with open(filename, 'r', encoding = 'utf8') as f:
        data, target = [], []
        for row in f.readlines():
            data.append(row.split('\t')[0])
            target.append(row.split('\t')[1].replace('\n', ''))
    return data, target

def encode_vocab(data, en):
    if en:
        tokenized = en_tokenize(data)
        data_vocab = sorted(list(set( [word.lower() \
                                       for words in tokenized \
                                       for word in words] )))
    else:
        tokenized = ch_tokenize(data)
        data_vocab = sorted(list(set( [word \
                                       for words in tokenized \
                                       for word in words] )))
    
    data_vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + data_vocab
    vocab_code = dict(zip(data_vocab, range(len(data_vocab))))
    vocab_size = len(vocab_code)
    encoded = []
    for sen in tokenized:
        temp_sen = []
        for word in sen:
            if en:
                temp_sen.append(vocab_code[word.lower()])
            else:
                temp_sen.append(vocab_code[word])  
        encoded.append(temp_sen)    
 
    return encoded, vocab_code, vocab_size

def padding(data, target):
    data_max_len = max([len(sen) for sen in data]) + 2 # add <START> and <END>
#    data_max_len = max([len(sen) for sen in data])
    for i, sen in enumerate(data):
        data[i] = [1] + data[i] + [2] # add <START> and <END>
        data[i] += [0]*(data_max_len - len(data[i]))
    target2 = list(target)  
    target_max_len = max([len(sen) for sen in target]) + 1 # add <START> or <END>
#    target_max_len = max([len(sen) for sen in target])
    for i, (sen, _) in enumerate(zip(target, target2)):
        target[i] = [1] + sen # add <START>
#        target[i] = sen
        target[i] += [0]*(target_max_len - len(target[i]))
        target2[i] = sen + [2] # add <END>  
#        target2[i] = sen[1:] + [2]
        target2[i] += [0]*(target_max_len - len(target2[i]))
    
    return data, target, target2, data_max_len, target_max_len
    
def choose_padding_len(data):
    data_len = sorted([len(sen) for sen in data])
    data_len = dict(Counter(data_len))
    l, c = zip(*data_len.items())
    plt.figure(figsize=(10,8))
    plt.plot(l,c)
    return c

#def load_vectors(vocab):
#    embeddings_index = {}
#    f = open('glove.6B.100d.txt', 'r')
#    for line in f:
#        values = line.split()
#        word = values[0]
#        if word in vocab:
#            coefs = np.asarray(values[1:], dtype='float32')
#            embeddings_index[word] = coefs
#    f.close()
#    print('Found %s word vectors.' % len(embeddings_index))

#%%

""" load data """
ori_data, ori_target = load_file('cmn.txt')

""" encode vocab """
data, en_code, en_vocab_size = encode_vocab(ori_data, True)
target, ch_code, ch_vocab_size = encode_vocab(ori_target, False)
de_ch_code = dict([(v, k) for k,v in ch_code.items()])

""" load pretrain glove weight """
weight_dim = 100
with open("glove_model.json", 'r') as f:
    glove_model = json.load(f)

glove_weight = [glove_model[word] \
                if word in glove_model.keys() \
                else [0.0]*weight_dim \
                for word in en_code]
glove_weight = np.array(glove_weight)

""" padding """
#count = choose_padding_len(data)
data, target, target2, input_size, output_size = padding(data, target)
data = np.array(data)
#data = data[:, ::-1]
target = np.array(target)
target2 = np.array(target2)
target2 = to_categorical(target2, num_classes=ch_vocab_size)

#%%

""" model """
## constant
batch_size = 64
epochs = 50
latent_dim = 256
num_samples = 20294

## encode
encoder_inputs = Input(shape=(input_size,))
encoder_embedding = Embedding(en_vocab_size, weight_dim, weights = [glove_weight], trainable = False)
encoded = encoder_embedding(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoded)
encoder_states = [state_h, state_c]

## decode
decoder_inputs = Input(shape=(output_size,))
decoder_embedding = Embedding(ch_vocab_size, weight_dim, trainable = True)
decoded = decoder_embedding(decoder_inputs)
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoded, initial_state=encoder_states)

## fully connected
dense = Dense(ch_vocab_size, activation='softmax')
decoder_outputs1 = dense(decoder_outputs)

## training model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([data, target], target2,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          verbose = 1)
model.save('s2s_'+mao+'.h5')



""" inference model """
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoded = decoder_embedding(decoder_inputs)
decoder_outputs, state_h, state_c = decoder(decoded, 
                                            initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs1 = dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs1] + decoder_states)

t = str(datetime.datetime.now())[:16]
encoder_model.save(t + '_encoder_model.h5')
decoder_model.save(t + '_decoder_model.h5')

encoder_model = load_model('2018-12-18 19:50_encoder_model.h5')
decoder_model = load_model('2018-12-18 19:50_decoder_model.h5')
#%%

""" test output """
test_index = [4077, 2122, 3335, 1464, 8956, 7168, 3490, 4495, 5100, 119]
outputs = ''

for index in test_index:
    stop_condition = False
    decoded_sentence = []
    
    test_data = data[index-1].reshape(1,-1)
    print('Input sentence:', ori_data[index-1])
    outputs += 'Input sentence: ' + ori_data[index-1] + '\n'
    
    states = encoder_model.predict(test_data)    
    target_seq = np.zeros((1, output_size))
    target_seq[0, 0] = 1 # first word is <START>
    
    word_n = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states)
#        states = [h, c]
        # Sample a token
        ch_index = np.argmax(output_tokens[0, word_n, :])
        ch_word = de_ch_code[ch_index]
        decoded_sentence.append(ch_word)

        # Exit condition: either hit max length
        # or find stop character.
        if ch_word == '<END>' or len(decoded_sentence)+1 > output_size:
            break

        # Update the target sequence (of length 1).
        word_n += 1
#        target_seq = np.zeros((1, output_size))
        target_seq[0, word_n] = ch_index # first word is ch_word

        # Update states
        
        
    print('Output sentence:', ' '.join(decoded_sentence))
    print('---')
    outputs += 'Output sentence: ' + ' '.join(decoded_sentence) + "\n---\n"
    
with open('outputs.txt', 'w') as f:
    f.write(outputs)


