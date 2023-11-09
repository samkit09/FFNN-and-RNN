import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, hidden = self.rnn(inputs)
        # [to fill] obtain output layer representations
        output = self.W(hidden)
        # [to fill] sum over output 
        out_sum = torch.sum(output, dim=0)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(out_sum)
        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # print("hidden dims \t= ",args.hidden_dim )
    # print("epochs \t= ",args.epochs )
    # print("train_data \t= ",args.train_data )
    # print("val_data \t= ",args.val_data )

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    print(len(train_data))

    print("========== Vectorizing data ==========") 
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open(os. getcwd()+'\\Data_Embedding\\word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    # print("RNN Example")

    # while not stopping_condition:
    #     random.shuffle(train_data)
    #     model.train()

    # rnn = nn.RNN(10, 20, 2)
    # print(rnn)
    # input = torch.randn(5, 3, 10)
    # print(input.size())
    # h0 = torch.randn(2, 3, 20)
    # print(h0.size())
    # output, hn = rnn(input, h0)
    # print("output -> ")
    # print(output)
    # print("hidden -> ")
    # print(hn)
        

    # python test.py -hd 4 -e 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json