# Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step1. Start.

step2. Import the required libraries.

step3. Upload the csv file and read the dataset.

step4. Check for any null values using the isnull() function.

step5. From sklearn.tree import DecisionTreeRegressor.

step6. Import metrics and calculate the Mean squared error.

step7. Apply metrics to the dataset, and predict the output.

step8. End.

## Program:
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by : Pranavesh Saikumar

RegisterNumber : 212223040149
*/
```
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2 = metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
<br><br><br>
## Output:

## MSE:

![image](https://github.com/user-attachments/assets/f262b6c0-0c16-46e0-b376-89be11acbde0)

## r2:

![image](https://github.com/user-attachments/assets/d9a2c1a3-4a1c-4418-8bf6-9230e4722e9a)

## Data Prediction:

![image](https://github.com/user-attachments/assets/869357a9-85a4-4056-92cb-0b38653663df)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
