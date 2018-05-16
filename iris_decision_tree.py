from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class IrisClassifier(object):
    def __init__(self):
        self.iris = load_iris() # wczytywanie danych
        self.names = self.iris.target_names
        self.decision_tree = tree.DecisionTreeClassifier() # utworzenie klasyfikatora
        # uczenie klasyfikatora przy uzyciu zbioru uczacego
        self.decision_tree.fit(self.iris.data, self.iris.target)

    def make_predictions(self, data):
        # sprawdzenie dzialania klasyfikatora na zbiorze testowym
        return self.decision_tree.predict(data)

    def info(self):
        print(self.iris.DESCR)
        

def main():

    iris_classifier = IrisClassifier()
    iris_classifier.info()
    
    # example
    sample = [[6.9, 3.1, 4.9, 1.5]]
    predictions = iris_classifier.make_predictions(sample)
    predictions_as_names = iris_classifier.names[predictions]
    print(predictions_as_names)

if __name__ == '__main__':
    main()