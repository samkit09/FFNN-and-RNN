# FFNN-and-RNN
 A Feedforward Neural Network (FFNN) and Recurrent Neural Network (RNN) for performing a 5-class Sentiment Analysis task

Make sure anaconda3 or miniconda3 is installed on your system.

Extract the contents of the file "assignment2.zip"

open terminal or command prompt/terminal in the extracted folder and type the following commands one by one.

1. conda create -n nlp python=3.8
2. conda activate nlp
3. conda install pip -y
4. pip install -r requirements.txt
4. Goto - https://pytorch.org/get-started/locally/ .Then select options as per your system to get command to install appropriate version of pytorch.
5. Copy the command and paste and run in command prompt/terminal.

6. While the directory FFNN-and-RNN where ffnn.py and rnn.py is present and with conda enironment activated, run following commands to run ffnn.py and rnn.py ->
	a. To run ffnn.py: python ffnn.py -hd 40 -e 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json
    b. To run rnn.py (without testing data): python rnn.py -hd 40 -e 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json
    c. To run rnn.py (with testing data): python rnn.py -hd 40 -e 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json
