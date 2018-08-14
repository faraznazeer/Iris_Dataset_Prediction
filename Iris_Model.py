from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np

#Loading Iris Dataset
iris = load_iris()

#Splitting Data
train_data = iris["data"][:135]
train_label = iris["target"][:135]

test_data = iris["data"][135:]
test_label = iris["target"][135:]

#Training The Model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_label)

print(" Accuracy : ",clf.score(test_data,test_label)*100)

#Prediction from user input
predict_input = input(" Enter sepal_length, sepal_width, petal_length, petal_width : ").split(",") 
predict_input = np.array(predict_input,dtype = float)

#Prediction Result
prediction = clf.predict(predict_input.reshape(1,-1))
print(" Species : Iris",iris["target_names"][int(prediction)])