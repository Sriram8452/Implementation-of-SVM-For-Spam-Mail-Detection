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
import pandas as pd
data = pd.read_csv("spam.csv", encoding = 'Windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size= 0.35, random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```


## Output:

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/1fede129-c612-4dbb-a95d-a347018b75b9)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/7c997a28-6c33-4c6d-ac91-965422955e74)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/50f4b335-4228-4eb1-997b-f911aa51922d)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/8102bbd0-2009-409e-821a-860df37fcb97)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/00d3b904-407d-4b00-a736-ad7df88c0f74)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/0fabe5c3-db80-40e8-b4c7-a1bf61d8d4b9)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/7064234d-efa9-4d30-b8db-833261005886)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/884d37ec-bd6f-44be-810f-2d450db99e08)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/09029386-6e18-42ca-b69d-e7fa0e3c51ba)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/53432554-ac29-4893-9afc-afb1f994e037)

![image](https://github.com/Sriram8452/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708032/551657ed-5e77-4d21-9b7e-f0307f0572e5)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
