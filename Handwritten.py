import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("C:/Users/Sarthak/Downloads/train.csv")

#print(data)

clf = DecisionTreeClassifier()

#Training Datasets

xtrain = dataset.iloc[0:21000,1:].values
train_label = dataset.iloc[0:21000,0].values

clf.fit(xtrain, train_label)

#Testing Data
xtest = dataset.iloc[21000:,1:].values
actual_label = dataset.iloc[21000:,0].values

#sample data
d = xtest[8] #can use any index below 42000
d.shape = (28,28)
plt.imshow(255-d,cmap = "gray") #we have 255-d because I want white background with black colour
plt.show()
print(clf.predict([xtest[8]]))

#accuracy
p = clf.predict([xtest]) #can't pass d because it only takes single row vector
count = 0
for i in range(0,21000):
   count += 1 
   if p[i]:
   	  print(actual_label[i])
   else:
      print("0")
   
   print("ACCURACY", (count/21000)*100)