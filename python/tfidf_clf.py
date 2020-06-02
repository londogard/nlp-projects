from pprint import pprint
from time import time
import logging

from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.metrics import classification_report

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

### Prepare Data: Perhaps only allow set amount of categories etc
dataset_path = '/home/londet/git/nlp-projects/datasets/swedish_tweet_combined.csv'
dataset = pd.read_csv(dataset_path)

random_state = 42
test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(dataset['raw'], dataset['y'], random_state=random_state, test_size=test_size)

### 

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1,2))),
    ('clf', SGDClassifier(alpha=1e-5,penalty='elasticnet')),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    # 'tfidf__max_df': (0.75),
    # 'tfidf__max_features': (None, 5000, 10000, 50000),
    #'tfidf__ngram_range': ( (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (20,50,80),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('elasticnet'),
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
    y_true, y_pred = y_test, grid_search.predict(X_test)
    print(classification_report(y_true, y_pred))

"""
Best score: 0.622
Best parameters set:
    clf__alpha: 1e-05
    clf__max_iter: 20
    clf__penalty: 'elasticnet'
    tfidf__max_df: 0.5
    tfidf__ngram_range: (1, 2)
"""