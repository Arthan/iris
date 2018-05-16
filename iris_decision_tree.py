from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# wczytywanie danych 
iris = load_iris()

# tablica z danymi
x = iris.data

# etykiety (prawidlowe odpowiedzi)
y = iris.target

# #getting label names i.e the three flower species
y_names = iris.target_names

# podzielenie danych na zbior uczacy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02)

# utworzenie klasyfikatora
decision_tree = tree.DecisionTreeClassifier()

# uczenie klasyfikatora przy uzyciu zbioru uczacego
decision_tree.fit(x_train, y_train)

# sprawdzenie dzialania klasyfikatora na zbiorze testowym
predictions = decision_tree.predict(x)

print (iris.DESCR)
print (y)
print (predictions)

#print (decision_tree.predict([x_train[0]]))
#print (y_train[0])
# na ile dokladny byl klasyfikator w przypadku zbioru testowego
output = accuracy_score(y, predictions) * 100
print(output)
