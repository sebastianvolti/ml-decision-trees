import math
import utils

def id3(data_set, attributes, gain_zero):
    positives, negatives = count_classes(data_set)
    if positives == len(data_set):
        node = {
            'attr': None,
            'values': 'positive'
        }
    elif negatives == len(data_set):
        node = {
            'attr': None,
            'values': 'negative'
        }
    elif len(attributes) == 0:
        if positives > negatives:
            node = {
                'attr': None,
                'values': 'positive'
            }
        else:
            node = {
                'attr': None,
                'values': 'negative'
            }
    else:
        total_examples, positive_value, negative_value = entropy_data_set(data_set)
        entropy_value = entropy(positive_value, negative_value, total_examples)
        best_attr = int(attributes[0])
        best_gain = gain(data_set, best_attr, entropy_value)
        for j in attributes:
            j = int(j)
            attr_gain = gain(data_set, j, entropy_value)
            if (attr_gain > best_gain):
                best_gain = attr_gain
                best_attr = j
        


        #Corto recursión, atributos ya no me aportan info          
        if(gain_zero == 1) and (best_gain == 0):

            if (positives >= negatives):
                value = 'positive'
            else:
                value = 'negative'

            node = {
                'attr': None,
                'values': value
            }  

        else:

            node = {
             'attr': best_attr,
             'values': {
                '0': {},
                '1': {}
              }
            }

            negative_data_set = filter_data_set(data_set, best_attr, 0)
            positive_data_set = filter_data_set(data_set, best_attr, 1)
            if len(negative_data_set) == 0:
                # Como solo hay 2 valores posibles para cada atributo, directamente el valor más probable es 0 o 1
                node['values']['0'] = {
                    'attr': None,
                    'values': 'positive'
                }
            else:
                attributes.remove(str(best_attr))
                node['values']['0'] = id3(negative_data_set, attributes, gain_zero)
                attributes.append(str(best_attr))

            if len(positive_data_set) == 0:
                node['values']['1'] = {
                    'attr': None,
                    'values': 'negative'
                }
            else:
                attributes.remove(str(best_attr))
                node['values']['1'] = id3(positive_data_set, attributes, gain_zero)
                attributes.append(str(best_attr))

    return node


def classify(tree, training_example):
    tree_value = tree['attr'];
    if (tree_value is None):
        return tree['values']
    else:
        current_node = tree['attr']
        current_train = training_example[current_node]
        return classify(tree['values'][str(current_train)], training_example)


def filter_data_set(data_set, attr, value):
    data_set_result = []
    for j in range (len(data_set)):
        if (str(data_set[j][attr]) == str(value)):
            data_set_result.append(data_set[j])
    return data_set_result


def count_classes(data_set):
    positives = 0
    negatives = 0
    for example in data_set:
        if example[-1] == 'positive':
            positives += 1
        elif example[-1] == 'negative':
            negatives += 1

    return positives, negatives


def amount_exaples(data_set,attr_idex):
    total_examples = len(data_set)
    exaples_with_value0 = []
    exaples_with_value1 = []
    for j in range (len(data_set)):
       
        if ((data_set[j][attr_idex]) == '0'):
            exaples_with_value0.append(data_set[j])
        elif ((data_set[j][attr_idex]) == '1') :
            exaples_with_value1.append(data_set[j])
    return  exaples_with_value0, exaples_with_value1

def gain(data_set,attr_idex,entropy_value):
    total_examples = len(data_set)
    attr_0,attr_1= amount_exaples(data_set,attr_idex)
    S0=(len(attr_0)/total_examples)
    
    total_examples0, positive_value0, negative_value0= entropy_data_set(attr_0)
    entropyS0=entropy(positive_value0, negative_value0,total_examples0)

    S1=(len(attr_1)/total_examples)
    total_examples1, positive_value1, negative_value1= entropy_data_set(attr_1)
    entropyS1=entropy(positive_value1, negative_value1,total_examples1)
    gain_value=entropy_value-((S0*entropyS0)+((S1)*entropyS1))
    # print('attr: ', attr_idex, 'cant 0:', len(attr_0), 'cant 1:', len(attr_1), 'entval: ', entropy_value, 'entS0: ', entropyS0, 'entS1: ', entropyS1, 'gain: ', gain_value)
    return gain_value


def entropy_data_set(data_set):
    total_examples = len(data_set)
    positive_value = 0
    negative_value = 0
    for j in range (len(data_set)):
        last_index = len(data_set[j]) - 1
        if (str(data_set[j][last_index]) == "positive"):
            positive_value+=1 
        else:
            negative_value+=1
    return total_examples, positive_value, negative_value


def entropy(positive_value, negative_value, total_examples):
    positive_result = 0
    negative_result = 0
    positive_coefficient = 0
    negative_coefficient = 0
    if (total_examples > 0):
        positive_coefficient = (positive_value/total_examples)
        negative_coefficient = (negative_value/total_examples)
    if (positive_coefficient != 0):
        positive_result = (-positive_coefficient)*(math.log2(positive_coefficient))
    if (negative_coefficient != 0):
        negative_result = (-negative_coefficient)*(math.log2(negative_coefficient))
    return (positive_result + negative_result)    


def classify_examples(tree, test_examples, test_classes):
    obtained_classes = []
    for example in test_examples:
        obtained_classes.append(classify(tree, example))

    acc = utils.get_accuracy(obtained_classes, test_classes)
    prec = utils.get_precision(obtained_classes, test_classes)
    rec = utils.get_recall(obtained_classes, test_classes)
    f1 = utils.get_f1_score(obtained_classes, test_classes)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'classification': obtained_classes # Ejemplos ya clasificados
    }

## Ejemplo de la estructura que usamos, se arma árbol clasificador de ejemplo de 3 nodos, llamado tree_ex
## Se trabaja con 2 ejemplos de entrenamiento de prueba, ubicados en training_set_example

def tree_example():

    tree_ex = {
        'attr': 0,
        'values': {
            '0': {
                'attr': 1,
                'values': {
                    '1': {
                        'attr': None,
                        'values': 0
                    },
                    '0': {
                        'attr': None,
                        'values': 0
                    }
                }
            },
            '1': {
                'attr': 2,
                'values': {
                    '1': {
                        'attr': None,
                        'values': 1
                    },
                    '0': {
                        'attr': None,
                        'values': 1
                    }
                }
            }
        }
    }

    training_set_example = []
    training_set_example.append([("att-0",0),("att-1",0),("att-2",0),("att-3",0),("att-4",0),("att-5",0),("att-6",0)])
    training_set_example.append([("att-0",1),("att-1",1),("att-2",1),("att-3",1),("att-4",1),("att-5",1),("att-6",1)])

    for i in range(len(training_set_example)):
        find_example(training_set_example[i], tree_ex)


def find_example(training_example, tree):

    tree_value = tree['attr'];
    if (tree_value is None):
        if (tree['values'] == 1):
            print("This example classification is POSITIVE!!!!!!!")
            print("Example", training_example)
        elif (tree['values'] == 0):
            print("This example classifsication is NEGATIVE!!!!!!!")
            print("Example", training_example)
    else:
        current_node = tree['attr']
        current_train = training_example[current_node][1]
        find_example(training_example, tree['values'][str(current_train)])
