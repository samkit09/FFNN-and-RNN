import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
# import math
import random
import os
# import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
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

def load_test_data(test_data):
    with open(test_data) as test:
        testing = json.load(test)
    tes = []
    for elt in testing:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))

    return tes

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    print("Length of training set: ",len(train_data))
    print("Length of validation set: ",len(valid_data))

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open(os. getcwd()+'\\Data_Embedding\\word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0
    training_losses = list()
    best_validation_accuracy = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None

            # loop for training model on train data 
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]
                vecs = np.array(vectors)

                # Transform the input into required shape
                vectors = torch.tensor(vecs).view(vecs.shape[0], 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        training_losses.append(loss_total)

        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        training_accuracy = correct/total

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        # loop for testing model on validation set and calculating accuracy
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]
            vecs = np.array(vectors)
            vectors = torch.tensor(vecs).view(vecs.shape[0], 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        if ((validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy) or (epoch == args.epochs)):
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", best_validation_accuracy)
        else:
            best_validation_accuracy = max(best_validation_accuracy,validation_accuracy)
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy

        epoch += 1

        if (args.test_data != "to fill") :
            test_data = load_test_data(args.test_data)
            correct = 0
            total = 0
            random.shuffle(test_data)
            valid_data = valid_data

            for input_words, gold_label in tqdm(test_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                vec = np.array(vectors)
                vectors = torch.tensor(vec).view(vec.shape[0], 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1

            print("\nTest accuracy for epoch {}: {}".format(epoch + 1, correct / total))

    # analysis: plotting training loss
    plt.plot(range(1,epoch+1), training_losses)
    plt.ylabel('Training Loss')
    plt.xlabel('Epochs')
    plt.show()

    # You may find it beneficial to keep track of training accuracy or training loss;
    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
    # https://youtu.be/WEV61GmmPrk

    # command to run code with best setting: python rnn.py -hd 40 -e 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json

