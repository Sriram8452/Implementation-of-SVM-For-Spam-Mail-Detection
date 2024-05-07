# Ex.No.9-Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of
SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: Sriram G
RegisterNumber:  212222230149
*/
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(10000))
result
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding="windows-1252")
```
```
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
x=data["v1"].values
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
```
```
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/001b72fe-8489-440e-8552-eb24ac186c93)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/0a6a7728-5df3-4b35-a130-7e6566dbfdc4)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/0e8f402f-9254-4039-ad96-54f7c121a784)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/ff1231e7-f081-4141-bbde-98472e76e443)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/4a603041-7b3a-4b0b-b565-2fd0afcf3269)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/5d0b61cb-05f5-49b5-bab2-41cd20832e24)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/aec5d4d5-8c30-4c94-90a9-ef9eb94c4ebe)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
