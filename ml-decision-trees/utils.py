import copy
import csv
import random
import math


def load_data(path):
    ''' Carga los datos de entrenamiento del .csv y lo transforma en una lista '''
    return list(csv.reader(open(path), delimiter=';'))


def partition_sets(data, training_set_percentage):
    ''' Revuelve los datos de entrenamiento y los separo en conjunto de entrenamiento y validación '''
    random.shuffle(data)
    training_set_size = int(len(data) * training_set_percentage)
    training_set = data[:training_set_size]
    test_set = data[training_set_size:]

    return training_set, test_set

def list_partition_sets(data, k):
    ''' Revuelve los datos de entrenamiento y los separo k conjuntos para validacion cruzada'''

#   Prueba para corroborar metodo de particionado
#
#   listaprueba = [1,2,3,4,5,6,7,8,9,10]
#   list_example_set = []
#   listapruebalargo = math.ceil(int((len(listaprueba)) * (1/10)))
#   print("listapruebalargo",listapruebalargo)
#   print("listaprueba",listaprueba)
#   for j in range(10):
#
#       lista1 = listaprueba[:listapruebalargo]
#       list_example_set.insert(j, lista1)
#       listaprueba = listaprueba[listapruebalargo:]
#       print("listaprueba", listaprueba)
#   print("lista prueba final",list_example_set) 

    random.shuffle(data)
    training_set_size = math.ceil(int((len(data)) * (1/k)))
    list_example_set = []
    for i in range(k):
        example_set = data[:training_set_size]
        list_example_set.insert(i, example_set)
        data = data[training_set_size:]
    return list_example_set




def get_classes(data):
    ''' Separa los datos de entrenamiento en dos listas distintas.
    Para cada ejemplo: [attr1, attr2, ..., attrN, class] en [attr1, attr2, ..., attrN] | [class] '''
    examples = copy.deepcopy(data)
    classes = []
    for example in examples:
        classes.append(example.pop(-1))

    return examples, classes


def count_equals(l1, l2, classification=None):
    ''' Cuenta los elementos iguales que hay en ambas listas, índice a índice.
        Si classification es None, cuenta las igualdades entre todos los elementos.
        Sino, si por ejemplo classification = 'positive' -> cuenta los elementos iguales entre
        ambas listas, pero que tienen valor 'positive' '''
    count = 0
    for i, elem in enumerate(l1):
        if elem == l2[i]:
            if classification is not None and elem == classification:
                count += 1
            elif classification is None:
                count += 1

    return count


def count_differences(l1, l2, classification=None):
    count = 0
    for i, elem in enumerate(l1):
        if elem != l2[i]:
            if classification is not None and elem == classification:
                count += 1
            elif classification is None:
                count += 1

    return count


def get_accuracy(obtained_classes, test_classes):
    hits = count_equals(obtained_classes, test_classes) # Obtengo los aciertos comparando ambas listas
    precision_base = len(obtained_classes)
    fraction = hits / precision_base
    return fraction


def get_precision(obtained_classes, test_classes):
    true_pos = count_equals(obtained_classes, test_classes, classification='positive') # Obtengo los verdaderos positivos comparando ambas listas
    precision_base = len([x for x in obtained_classes if x == 'positive']) # Cantidad de clasificaciones positivas
    if precision_base==0:
        return 0
    else:
        fraction = true_pos / precision_base
        return fraction


def get_recall(obtained_classes, test_classes):
    true_pos = count_equals(obtained_classes, test_classes, classification='positive') # Obtengo los verdaderos positivos comparando ambas listas
    false_neg = count_differences(obtained_classes, test_classes, classification='negative')
    if false_neg==0:
        return 0
    else :
        fraction = true_pos / (true_pos + false_neg)
        return fraction


def get_f1_score(obtained_classes, test_classes):
    prec = get_precision(obtained_classes, test_classes)
    rec = get_recall(obtained_classes, test_classes)
    if prec==0 and rec==0:
        return 0 
    else:
        return (2 * prec * rec) / (prec + rec)


def analize_results(tree, test_examples, test_classes):
    negative_leaves = examinate_tree(tree, 'positive')
    positive_leaves = examinate_tree(tree, 'negative')
    error_negative, error_positive = analize_evaluation(tree, test_examples, test_classes)
    print("\n")
    print("Resultados obtenidos:")
    print(" =>")
    print("Detalles de árbol clasificador generado:")
    print("Cantidad de hojas de árbol clasificador:", negative_leaves + positive_leaves)
    print("Cantidad de hojas con valor positivo:", negative_leaves)
    print("Cantidad de hojas con valor negativo:", positive_leaves)
    print("\n")
    print("Resultados de evaluación sobre test_set:")
    print(" =>")
    print("Cantidad de ejemplos con valor NEGATIVO clasificados MAL:" , error_negative)
    print("Cantidad de ejemplos con valor POSITIVO clasificados MAL:" , error_positive)



def analize_evaluation(tree, test_examples, test_classes):
    error_positive = 0
    error_negative = 0
    examples_classes = []
    for example in test_examples:
        examples_classes.append(classify_utils(tree, example))

    for i, elem in enumerate(examples_classes):
        if elem != test_classes[i]:
            if (test_classes[i] == "positive"):
                error_positive+=1
            else:
                error_negative+=1
    return error_negative, error_positive

def analize_evaluation_RF(forest, test_examples, test_classes):
    error_positive = 0
    error_negative = 0
    examples_classes = []
    for example in test_examples:
        examples_classes.append(forest.Evaluation(example))

    for i, elem in enumerate(examples_classes):
        if elem != test_classes[i]:
            if (test_classes[i] == "positive"):
                error_positive+=1
            else:
                error_negative+=1
    return error_negative, error_positive

def analize_results_RF(forest, test_examples, test_classes):
    error_negative, error_positive = analize_evaluation_RF(forest, test_examples, test_classes)
    print("\n")
    print("Resultados de evaluación sobre test_set:")
    print(" =>")
    print("Cantidad de ejemplos con valor NEGATIVO clasificados MAL:" , error_negative)
    print("Cantidad de ejemplos con valor POSITIVO clasificados MAL:" , error_positive)


def classify_utils(tree, training_example):
    tree_value = tree['attr'];
    if (tree_value is None):
        return tree['values']
    else:
        current_node = tree['attr']
        current_train = training_example[current_node]
        return classify_utils(tree['values'][str(current_train)], training_example)

    
def analize_data_set(training_set, test_set, percentage):
    positives_train = 0
    positives_test = 0
    negatives_train = 0
    negatives_test = 0

    for example in training_set:
        if example[-1] == 'positive':
            positives_train += 1
        elif example[-1] == 'negative':
            negatives_train += 1

    for example in test_set:
        if example[-1] == 'positive':
            positives_test += 1
        elif example[-1] == 'negative':
            negatives_test += 1   

    print("ANÁLISIS DE LA DISTRUBUCIÓN DE LOS EJEMPLOS")
    print(" =>")
    print("Porcentaje de ejemplos utilizados para training_set:", percentage*100)
    print("Porcentaje de ejemplos utilizados para test_set:", (1-percentage)*100)
    print("\n")
    print("Ejemplos de entrenamiento:")
    print(" =>")
    print("Total de ejemplos en training_set: ", negatives_train + positives_train)
    print("Ejemplos Positivos en training_set: ", positives_train)        
    print("Ejemplos Negativos en training_set: ", negatives_train)    
    print("Porcentaje de ejemplos negativos: ",(negatives_train/(negatives_train + positives_train))*100)
    print("Porcentaje de ejemplos positivos: ",(positives_train/(negatives_train + positives_train))*100)
    print("\n")
    print("Ejemplos de evaluación:")   
    print(" =>")
    print("Total de ejemplos en test_set: ", positives_test + negatives_test)
    print("Ejemplos Positivos en test_set: ", positives_test)        
    print("Ejemplos Negativos en test_set: ", negatives_test)    
    print("Porcentaje de ejemplos negativos: ",(negatives_test/(negatives_test + positives_test))*100)
    print("Porcentaje de ejemplos positivos: ",(positives_test/(negatives_test + positives_test))*100)
    print("----------------------------------------")
    print()
    print()


def examinate_tree(tree, value):
    tree_value = tree['attr']
    if (tree_value is None):
        if (tree['values'] == value):
            return 1
        else:
            return 0
    else:
        current_node = tree['attr']
        return examinate_tree(tree['values']['0'], value) + examinate_tree(tree['values']['1'], value)


def examinate_depth(tree, depth, list_depth):
    tree_value = tree['attr']
    if (tree_value is None):
       list_depth.append(depth)
    else:
        current_node = tree['attr']
        examinate_depth(tree['values']['0'], 1 + depth, list_depth) 
        examinate_depth(tree['values']['1'], 1 + depth, list_depth)