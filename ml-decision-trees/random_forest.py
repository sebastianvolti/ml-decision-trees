import logging
import random
import statistics
import utils
import math 
import id3
import random
import time

class RandomForest(object):
    def __init__(self,k,m,j,training_set_original,gain_zero,use_sqrt):
        self.forest=[]
        training_set=[]
        if use_sqrt:
            l=math.ceil(math.sqrt(j))
        else:
            l=math.ceil(j/3)

        for i in range(m):
            training_set=[]
            
            # Generar el conjunto Di sacando k elementos de D con repeticion 
            for t in range(k):
                training_set.append(training_set_original[random.randrange(k)])
            # Generar cantidad de atribitos  l aleatorios
            h=random.sample(range(0,j),l)
            attributes = [str(i) for i in h]

            logging.debug(f"Construcción id3 {i + 1} de {m}")
            start_time = time.time()
            tree=id3.id3(training_set,attributes,gain_zero)
            logging.debug("--- {:.2f} segundos ---".format(time.time() - start_time))
            self.forest.append(tree)


    def Evaluation(self,data):
        positive=0
        negatives=0
        c=0
        for i in range(len(self.forest)):
            tree=self.forest[i]
            c=id3.classify(tree,data)#Classify no devuelve nada
            if c=='negative':
                negatives+= 1
            elif c=='positive':
                positive+= 1
        if negatives<positive:
            return 'positive'
        else:
            return 'negative'
    

    def Evaluation_testset(self, test_examples,test_classes):
        obtained_classes = []
        for example in test_examples:
            obtained_classes.append(self.Evaluation(example))

        acc = utils.get_accuracy(obtained_classes, test_classes)
        prec = utils.get_precision(obtained_classes, test_classes)
        rec = utils.get_recall(obtained_classes, test_classes)
        f1 = utils.get_f1_score(obtained_classes, test_classes)

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            #'classification': obtained_classes # Ejemplos ya clasificados
    }
