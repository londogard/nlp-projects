from pprint import pprint
from time import time
import logging

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from gensim.models.fasttext import load_facebook_vectors

from fse.models import uSIF
from fse import SplitIndexedList

from fse.models.average import FAST_VERSION, MAX_WORDS_IN_BATCH
print(MAX_WORDS_IN_BATCH)
print(FAST_VERSION)

import pandas as pd

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

### Load Word Embedding models
# model_path = '/home/londet/git/nlp-projects/models/cc.sv.100.bin'
# embedding_model = load_facebook_vectors(model_path)

### Prepare Data: Perhaps only allow set amount of categories etc
dataset_path = '/home/londet/git/nlp-projects/datasets/swedish_tweet_combined.csv'
dataset = pd.read_csv(dataset_path)
splitindexlist = SplitIndexedList(dataset['raw'].tolist())

#model = uSIF(embedding_model, workers=2, lang_freq="sv")
#model.train(splitindexlist)
#model.save('/home/londet/git/nlp-projects/models/sfe.bin')
model = uSIF.load('/home/londet/git/nlp-projects/models/sfe.bin')
# dataset['X'] = dataset['raw'].apply(lambda x: )

random_state = 42
test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(model.sv, dataset['y'], random_state=random_state, test_size=test_size)


# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([('clf', LinearSVC())])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    # 'clf__max_iter': (20,),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

""" (0.57 with all examples)
Best score: 0.531
Best parameters set:
    clf__alpha: 1e-05
    clf__max_iter: 20
    clf__penalty: 'elasticnet'
"""