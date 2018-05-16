import unittest
from random import shuffle
from sklearn.metrics import accuracy_score
from iris_decision_tree import IrisClassifier

class TestIrisClassifier(unittest.TestCase):
    def setUp(self):
        self.iris_classifier = IrisClassifier()
        
    def test_random_10(self):
        test_ids = list(range(len(self.iris_classifier.iris.data)))
        shuffle(test_ids)
        test_ids = test_ids[:10]
        
        data = self.iris_classifier.iris.data[test_ids]
        labels = self.iris_classifier.iris.target[test_ids]
        predictions = self.iris_classifier.make_predictions(data)
        accuracy = accuracy_score(labels, predictions) * 100
        #print(labels)
        #print(predictions)
        #print(accuracy)
        self.assertEqual(accuracy, 100.0)
        
if __name__ == '__main__':
    unittest.main()