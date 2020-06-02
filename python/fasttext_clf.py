from pprint import pprint
from time import time
import logging

from sklearn.model_selection import train_test_split
import pandas as pd
import fasttext

### Prepare Data: Perhaps only allow set amount of categories etc
dataset_path = '/home/londet/git/nlp-projects/datasets/swedish_tweet_combined.csv'
dataset = pd.read_csv(dataset_path)

random_state = 42
test_size = 0.3

X = dataset['raw'].tolist()
y = [f'__label__{x}' for x in dataset['y'].tolist()]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)


open('text.in', 'w').write('\n'.join([f'{y} {x}' for x,y in zip(X_train,y_train)]))
open('text.valid', 'w').write('\n'.join([f'{y} {x}' for x,y in zip(X_test,y_test)]))

# model = fasttext.load_model('/home/londet/git/nlp-projects/models/cc.sv.100.bin')
model = fasttext.train_supervised(input='text.in',lr=1.0, epoch=25, wordNgrams=2)
# model.save_model
print(model.test('text.valid'))

# 0.622