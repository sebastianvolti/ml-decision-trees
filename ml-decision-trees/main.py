import argparse
import logging
import numpy as np
import random
import pprint
import statistics
import utils
import sys
import time

import id3
import random_forest
import scikit


################## Config básica ##################

DEBUG = True

# Lee argumentos
ap = argparse.ArgumentParser(
    description='Tarea de AprendAut')
ap.add_argument('-s', '--seed', help='Indica la semilla a utilizar para la librería random')
ap.add_argument('-t', '--tset_percentage', help='Indica con un valor entre 0 y 1, qué porcentaje de los datos se usan para el entrenamiento (el resto es para validación')
ap.add_argument('-p', '--part', help='Indica con valores a, b o c, qué parte de la tarea se quiere ejecutar')
ap.add_argument('-d', '--debug_level', default=2, help='0 si no se quiere imprimir nada, 1 para mensajes de info, 2 para mensajes de debug')
ap.add_argument('-m', '--m_rf', default=100, help='Valor de m (cantidad de árboles) para realizar el random forest. Por defecto es 100')
ap.add_argument('-q', '--use_sqrt', default=0, help='Indica con un 1 si se usa la raíz cuadrada de k para la cantidad de atributos, o si se usa j/3 (con un 0)')
ap.add_argument('-r', '--r_print', default=0, help='Si r = 1 se imprime analisis de resultados.')
ap.add_argument('-g', '--gain_zero', default=0, help='Si g = 1 se corta recursión en ID3 cuando máxima ganacia de attrs = 0.')
ap.add_argument('-b', '--balanced_ds', default=0, help='Si b = 1 se trabaja con un data set balanceado, (741+, 741-).')
ap.add_argument('-c', '--cross_validation', default=0, help='Si c > 0 se utiliza validación cruzada de "c" iteraciones')


args = vars(ap.parse_args())
seed = int(args['seed'])
training_set_percentage = float(args['tset_percentage'])
part = args['part']
debug_level = int(args['debug_level'])
m = int(args['m_rf'])
use_sqrt = int(args['use_sqrt'])
show_results = int(args['r_print'])
gain_zero = int(args['gain_zero'])
balanced = int(args['balanced_ds'])
cross_validation = int(args['cross_validation'])


if debug_level == 0:
    logging_level = logging.WARNING
elif debug_level == 1:
    logging_level = logging.INFO
elif debug_level == 2:
    logging_level = logging.DEBUG

logging.basicConfig(level=logging_level, format='%(message)s')


def print_metrics(data):
    logging.info(f"Accuracy: {data['accuracy']*100:.2f}%")
    logging.info(f"Precision: {data['precision']*100:.2f}%")
    logging.info(f"Recall: {data['recall']*100:.2f}%")
    logging.info(f"F1: {data['f1']*100:.2f}%")

    if (show_results == 1):
        print(f"Accuracy: {data['accuracy']*100:.2f}%")
        print(f"Precision: {data['precision']*100:.2f}%")
        print(f"Recall: {data['recall']*100:.2f}%")
        print(f"F1: {data['f1']*100:.2f}%")



################## Comienzo del main ##################

def main():
    global m, use_sqrt

    sys.setrecursionlimit(3000)
    random.seed(seed)
    np.random.seed(seed) # Seteo la seed para numpy (que es lo que usa scikit)
    data = utils.load_data('qsar_oral_toxicity.csv')

    if (cross_validation > 0):
        
        #Particion de data set para cross_validation
        list_example_set = utils.list_partition_sets(data, cross_validation)
        acc_prom = 0
        pre_prom = 0
        rec_prom = 0
        f1_prom  = 0
        acc_prom_sk = 0
        pre_prom_sk = 0
        rec_prom_sk = 0
        f1_prom_sk  = 0

        for i in range(len(list_example_set)):
            test_set = []
            training_set = []
            test_set = list_example_set[i]
            for j in range(len(list_example_set)):
                if (j != i):
                    training_set = training_set+list_example_set[j]

            if (show_results == 1):
                utils.analize_data_set(training_set, test_set, training_set_percentage)
                    
            # Separo los conjuntos en atributos | clase
            training_examples, training_classes = utils.get_classes(training_set)
            test_examples, test_classes = utils.get_classes(test_set)

            # Defino los atributos iniciales para el algoritmo ID3
            initial_attributes = [str(i) for i in range(1024)]

            classifier_id3 = id3.id3(training_set, initial_attributes, gain_zero)
            classifier_id3_sk = scikit.scikit_id3(training_examples, training_classes)

            classification_id3 = id3.classify_examples(classifier_id3, test_examples, test_classes)
            #print_metrics(classification_id3)
            
            acc_prom += classification_id3['accuracy']
            pre_prom += classification_id3['precision']
            rec_prom += classification_id3['recall']
            f1_prom  += classification_id3['f1']
            
            classification_id3_sk = scikit.classify_examples(classifier_id3_sk, test_examples, test_classes)
            #print_metrics(classification_id3_sk)
            
            acc_prom_sk += classification_id3_sk['accuracy']
            pre_prom_sk += classification_id3_sk['precision']
            rec_prom_sk += classification_id3_sk['recall']
            f1_prom_sk  += classification_id3_sk['f1']
            

        acc_prom = acc_prom/cross_validation
        pre_prom = pre_prom/cross_validation
        rec_prom = rec_prom/cross_validation
        f1_prom  = f1_prom/cross_validation

        acc_prom_sk = acc_prom_sk/cross_validation
        pre_prom_sk = pre_prom_sk/cross_validation
        rec_prom_sk = rec_prom_sk/cross_validation
        f1_prom_sk  = f1_prom_sk/cross_validation


        print("Resultados cross validation id3 nuestro:")   
        print(" =>")
            
        print(f"Accuracy: {acc_prom*100:.2f}%")
        print(f"Precision: {pre_prom*100:.2f}%")
        print(f"Recall: {rec_prom*100:.2f}%")
        print(f"F1: {f1_prom*100:.2f}%")



        print("Resultados cross validation id3 scikit:")   
        print(" =>")
            
        print(f"Accuracy: {acc_prom_sk*100:.2f}%")
        print(f"Precision: {pre_prom_sk*100:.2f}%")
        print(f"Recall: {rec_prom_sk*100:.2f}%")
        print(f"F1: {f1_prom_sk*100:.2f}%")


    else:
        if (balanced == 1):
            balanced_data_set = [] 
            counter_limit = 741
            for elem in data:
                if (elem[1024] == 'negative'):
                    counter_limit-=1
                    if (counter_limit > 0):
                        balanced_data_set.append(elem)
                else:
                    balanced_data_set.append(elem)
            # Separo los conjuntos en entrenamiento | validación
            training_set, test_set = utils.partition_sets(balanced_data_set, training_set_percentage)
        else:
            # Separo los conjuntos en entrenamiento | validación
            training_set, test_set = utils.partition_sets(data, training_set_percentage)

        if (show_results == 1):
            utils.analize_data_set(training_set, test_set, training_set_percentage)


        # Separo los conjuntos en atributos | clase
        training_examples, training_classes = utils.get_classes(training_set)
        test_examples, test_classes = utils.get_classes(test_set)

        # Defino los atributos iniciales para el algoritmo ID3
        initial_attributes = [str(i) for i in range(1024)]

        if part == 'a':

            logging.info(f"Construyendo árbol con ID3 nuestro...")
            tree = id3.id3(training_set, initial_attributes, gain_zero)
            logging.info(f"ID3 nuestro construido")
            classification_id3 = id3.classify_examples(tree, test_examples, test_classes)
            logging.info(f"Resultados ID3 nuestro")
            print_metrics(classification_id3)

            if (show_results == 1):
                utils.analize_results(tree, test_examples, test_classes)


        elif part == 'b':
            k_rf = len(training_set) # k = |D|
            logging.info(f"Construyendo Random Forest nuestro...")
            rf = random_forest.RandomForest(k_rf,m,1024,training_set,gain_zero,use_sqrt)
            logging.info(f"Random Forest nuestro construido")
            classification_rf = rf.Evaluation_testset(test_examples,test_classes)
            logging.info(f"Resultados Random Forest nuestro")
            print_metrics(classification_rf)

            if (show_results == 1):
                utils.analize_results_RF(rf, test_examples, test_classes)

        elif part == 'c':
            ######### ID3 #########
            # Construyo los árboles de decisión con los ejemplos de entrenamiento
            logging.info("CONSTRUCCIÓN DE ÁRBOLES CON ID3")
            logging.info(" =>")
            logging.info(f"Construyendo árbol con ID3 nuestro...")
            start_time = time.time()
            classifier_id3 = id3.id3(training_set, initial_attributes, gain_zero)
            logging.info(f"ID3 nuestro construido")
            logging.debug("--- {:.2f} segundos ---".format(time.time() - start_time))

            classifier_id3_sk = scikit.scikit_id3(training_examples, training_classes)
            logging.info(f"ID3 de scikit construido")
            logging.info("----------------------------------------")
            logging.info("")
            logging.info("")

            # Evalúo los ID3 usando los ejemplos de validación
            logging.info("RESULTADOS COMPARACIÓN ID3")
            logging.info(" =>")
            classification_id3 = id3.classify_examples(classifier_id3, test_examples, test_classes)
            logging.info(f"Resultados ID3 nuestro")
            print_metrics(classification_id3)
            logging.info("")

            classification_id3_sk = scikit.classify_examples(classifier_id3_sk, test_examples, test_classes)
            logging.info(f"Resultados ID3 de scikit")
            print_metrics(classification_id3_sk)
            logging.info("----------------------------------------")
            logging.info("")
            logging.info("")

            ######### Random forest #########
            # Construyo los random forests con los ejemplos de entrenamiento
            logging.info("CONSTRUCCIÓN DE RANDOM FORESTS")
            logging.info(" =>")
            k_rf = len(training_set) # k = |D|
            logging.info(f"Construyendo Random Forest nuestro...")
            classifier_rf = random_forest.RandomForest(k_rf,m,1024,training_set,gain_zero,use_sqrt)
            logging.info(f"Random Forest nuestro construido")

            classifier_rf_sk = scikit.scikit_rf(training_examples, training_classes, m, use_sqrt)
            logging.info(f"Random Forest de scikit construido")
            logging.info("----------------------------------------")
            logging.info("")
            logging.info("")

            # Evalúo el random forest usando los ejemplos de validación
            logging.info("RESULTADOS COMPARACIÓN RANDOM FORESTS")
            logging.info(" =>")
            classification_rf = classifier_rf.Evaluation_testset(test_examples,test_classes)
            logging.info(f"Resultados Random Forest nuestro")
            print_metrics(classification_rf)
            logging.info("")

            classification_rf_sk = scikit.classify_examples(classifier_rf_sk, test_examples, test_classes)
            logging.info(f"Resultados Random Forest de scikit")
            print_metrics(classification_rf_sk)
            logging.info("----------------------------------------")
            logging.info("")
            logging.info("")

        elif part == 'g':
            import plots

            use_sqrt = 0
            k_rf = len(training_set) # k = |D|
            ms = [1, 2, 5, 10, 20, 40, 65, 100]
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            results = {
                'ours': {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                },
                'scikit': {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                }
            }

            for m in ms:
                classifier_rf = random_forest.RandomForest(k_rf,m,1024,training_set,gain_zero,use_sqrt)
                classification_rf = classifier_rf.Evaluation_testset(test_examples,test_classes)

                classifier_rf_sk = scikit.scikit_rf(training_examples, training_classes, m, use_sqrt)
                classification_rf_sk = scikit.classify_examples(classifier_rf_sk, test_examples, test_classes)

                for metric in metrics:
                    results['ours'][metric].append(classification_rf[metric] * 100)
                    results['scikit'][metric].append(classification_rf_sk[metric] * 100)

            for metric in metrics:
                plots.plot(ms, results['ours'][metric], results['scikit'][metric], metric)


if __name__ == "__main__":
   main()
