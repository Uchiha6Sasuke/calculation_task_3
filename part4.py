import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('Dataframe.csv', sep=' ')
dataset = dataset.drop(dataset[dataset.iris_species == 'virginica'].index)
X = dataset['sepal_length'].values.reshape(-1, 1)
y = dataset['iris_species'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=27)
classifier = GaussianNB()
# тренируем классификатор
classifier.fit(X_train, y_train)
# предсказываем
y_pred = classifier.predict(X_test)
print('4')
print('Пример предсказания класса по длине чашелистика:')
print('При длине чaшелистика 2.7 класс: ', ''.join(classifier.predict([[2.7]])))
print('При длине чашелистика 3.5 класс: ', ''.join(classifier.predict([[3.5]])))
print("Accuracy : ", accuracy_score(y_test, y_pred))
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print('Пример:')
print(df)