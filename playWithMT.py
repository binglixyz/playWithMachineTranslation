# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:47:55 2017

@player: bingl
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import time
import math

use_cuda = torch.cuda.is_available()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
#print(unicodeToAscii("I love you"))
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
#print(normalizeString(".!?"))

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

#print(len(readLangs("eng","fra")[2]))


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)



def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#pairs = readLangs("english","franch", reverse=True)[2]
#print(len(pairs))
#print(len(filterPairs(pairs)))

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

input_lang, output_lang, pairs = prepareData("eng","fra", True)

hidden_size = 256
MAX_LENGTH = 1000
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = DecoderRNN(hidden_size, output_lang.n_words)
    
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100):
    start = time.time()
    
    criterion = nn.NLLLoss()
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    loss = 0
    
    plot_loss_total = 0
    plot_losses = []
    
    print_loss_total = 0
    
    if n_iters != 0:
        training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
    else:
        training_pairs = [variablesFromPair(pair) for pair in pairs]
        n_iters = len(training_pairs)
        
    iter_count = 0
    for training_pair in training_pairs :
        input_variable = training_pair[0]
        output_variable = training_pair[1]
        loss = train(input_variable, output_variable, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer)
        iter_count += 1
        print_loss_total += loss
        plot_loss_total += loss
        if iter_count % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        if iter_count % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter_count / n_iters),
                                         iter_count, iter_count / n_iters * 100, print_loss_avg))
    showPlot(plot_losses)
    
teacher_forcing_ratio = 0
def train(input_variable, target_variable, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    loss = 0
    for word_idx in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[word_idx], encoder_hidden)
        
    decoder_hidden = encoder_hidden
    #print(encoder_hidden)
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for word_idx in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[word_idx])
            decoder_input = target_variable[word_idx]
    else:
        for word_idx in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[word_idx])
            predicted_prob, predicted_idx = decoder_output.data.topk(1)
            predicted_word = predicted_idx[0][0]
            decoder_input = Variable(torch.LongTensor([[predicted_word]]))
            if predicted_word == EOS_token:
                break;
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length
    

def evaluate(encoder, decoder, sentence, max_length = MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence);
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    for word_idx in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[word_idx], encoder_hidden)
    decoder_hidden = encoder_hidden
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoded_words = []
    for word_idx in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        next_word_prob, next_word_idx = decoder_output.data.topk(1)
        next_word = next_word_idx[0][0]
        decoder_input = Variable(torch.LongTensor([[next_word]]))
        if next_word == EOS_token:
            decoded_words.append("<EOS>")
            break;
        else:
            decoded_words.append(output_lang.index2word[next_word])
    return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

#encoder = EncoderRNN(input_lang.n_words, hidden_size)
#encoder.load_state_dict(torch.load("./model/encoder.model"))
trainIters(encoder, decoder, 0);
torch.save(encoder.state_dict(), "./model/encoder.model")
torch.save(decoder.state_dict(), "./model/encoder.model")
evaluateRandomly(encoder, decoder)
input("?")