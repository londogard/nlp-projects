from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings, FastTextEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md

corpus = ClassificationCorpus(Path('./'),
                              test_file='text.valid',
                              train_file='text.in')
word_embeddings = [FastTextEmbeddings('cc.sv.100.bin'), FlairEmbeddings('sv-forward'), FlairEmbeddings('sv-backward')]
document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)
classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
trainer = ModelTrainer(classifier, corpus)
trainer.train('./', max_epochs=10)
# TODO get this to work!
# https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f

"""
MICRO_AVG: acc 0.693384223918575 - f1-score 0.5400763358778626
MACRO_AVG: acc 0.6933842239185749 - f1-score 0.5367466174866072
"""