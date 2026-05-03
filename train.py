import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("student.csv")
clear=df.drop(columns=["city"])

#assing variables
x=clear[["study_hours","attendance","sleep_hours"]]
y=clear["marks"]

#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#train data
model=LinearRegression()
model.fit(x_train,y_train)
joblib.dump(model,"student_model.pkl")