import math

from sklearn import tree, ensemble
from sklearn.model_selection import cross_val_score

import id3
import utils


# ID3

def scikit_id3(examples, classes):
    ''' Construye el arbol de decisión con ID3 utilizando sklearn '''
    classifier = tree.DecisionTreeClassifier(criterion='entropy')
    classifier = classifier.fit(examples, classes)

    return classifier


# Random forest

def scikit_rf(examples, classes, m, use_sqrt):
    ''' Construye los árboles de decisión del random forest utilizando sklearn '''
    total_attrs = len(examples[0])
    if use_sqrt:
        attr_count = math.ceil(math.sqrt(total_attrs))
    else:
        attr_count = math.ceil(total_attrs / 3)

    classifier = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=m, max_features=attr_count)
    classifier = classifier.fit(examples, classes)

    return classifier


# Misc

def classify(classifier, example):
    return classifier.predict([example])
    

def classify_examples(classifier, test_examples, test_classes):
    obtained_classes = list(classifier.predict(test_examples))
    # accuracy = classifier.score(test_examples, test_classes)
    acc = utils.get_accuracy(obtained_classes, test_classes)
    prec = utils.get_precision(obtained_classes, test_classes)
    rec = utils.get_recall(obtained_classes, test_classes)
    f1 = utils.get_f1_score(obtained_classes, test_classes)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'classification': obtained_classes # Clasificación de ejemplos
    }


def cross_validation_scores(classifier, test_examples, test_classes, cv=5):
    return cross_val_score(classifier, test_examples, test_classes, cv=cv)

