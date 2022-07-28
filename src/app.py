from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

url='https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
df=pd.read_csv(url)

df = df[(df['Glucose'] > 0) & (df['BloodPressure'] > 0) & (df['SkinThickness'] > 0) & (df['Insulin'] > 0)& (df['BMI'] > 0)]
df[df['Pregnancies']>11]
df=df.drop('Pregnancies',axis=1)
scale= StandardScaler()
df_st = scale.fit_transform(df) 
df_st=pd.DataFrame(df_st)
df_st.columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction','Age', 'Outcome']
x=df_st.drop('Outcome',axis=1)
y=df['Outcome']
X_train, X_test, y_train, y_test=train_test_split(x,y,stratify=y,random_state=34)

clf = DecisionTreeClassifier(criterion='entropy',random_state=42,max_leaf_nodes=2)
clf.fit(X_train, y_train)
clf_pred=clf.predict(X_test)