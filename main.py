from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.write("""
# Car Price Prediction
""")

df = pd.read_csv('data.csv')
st.sidebar.title('Navigation')
rad = st.sidebar.radio(
    '', ["Home", "Dataset (after cleaning)", "Predictor"])

st.write(f"## {rad} ")
if rad == "Home":
    st.write("""
    In this project we will be analyzing the [dataset taken from kaggle](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho) and predict the selling price of a car,
    by creating various regression model like mulitple linear regression, SVR, decision tree regression and random forest regression.
    """)

    st.header("Correlation between different variables")

    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(df.corr(), annot=True, cmap='mako', linewidths=.5)
    st.pyplot(fig)
    st.write("""
    So we can see a strong relation between **selling price** and **max_power_bhp**, and
    a moderate correlation of **selling price** with **engine_cc**.
    """)

    def reg_select(reg_name):
        reg = None
        if reg_name == 'Multiple Linear Regression':
            reg = LinearRegression()
        elif reg_name == 'Decision Tree Regression':
            reg = DecisionTreeRegressor()
        elif reg_name == 'SVR(Support Vector Regression)':
            ker = st.selectbox('Select the kernel', [
                               'poly', 'rbf', 'sigmoid'], 1)
            print(ker)
            reg = SVR(kernel=ker)
        elif reg_name == 'Random Forest Regression':
            esti = st.slider('Select the number of estimators',
                             min_value=100, max_value=500, value=100)
            reg = RandomForestRegressor(n_estimators=esti)
        return reg

    # Modelling
    st.header("Modelling")
    st.subheader('Select the regression model')
    select_reg = st.selectbox('Select regressor',
                              ['Multiple Linear Regression', 'SVR(Support Vector Regression)',
                               'Decision Tree Regression', 'Random Forest Regression'],
                              3)

    st.subheader('Set the parameter')
    split_ratio = st.slider(
        f'Data split ratio (% for Training Set)', min_value=10, max_value=90, step=10, value=80)/100
    # # Importing the dataset

    data = pd.read_csv('data.csv')
    data.head()

    x = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # # Encoding categorical variables

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [
        2, 3, 4, 5])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))

    print(x[:2])
    print(y[:2])

    # # Splitting the dataset into training and test set

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=split_ratio, random_state=0)

    # # Training the random forest regression model on the training set

    regressor = reg_select(select_reg)

    if(select_reg == 'SVR(Support Vector Regression)'):
        # feature scaling
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        x_train[:, 14:-1] = sc_x.fit_transform(x_train[:, 14:-1])
        y_train = sc_y.fit_transform(y_train.reshape(len(y_train), 1))

        regressor.fit(x_train, y_train.ravel())

        x_test[:, 14:-1] = sc_x.transform(x_test[:, 14:-1])

        y_pred_test = sc_y.inverse_transform(regressor.predict(x_test))
        y_pred_train = sc_y.inverse_transform(
            regressor.predict(x_train))
        y_train = sc_y.inverse_transform(y_train)
    else:
        regressor.fit(x_train, y_train)

        # # Predicting the test set values

        y_pred_test = regressor.predict(x_test)
        y_pred_train = regressor.predict(x_train)

    # # Evaluating the model performance

    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    st.write(f'**Shape of dataset**')
    st.info(data.shape)

    st.subheader('**Training set**')
    st.write(f'**$R^2$(coefficient of determination)**')
    st.info(r2_train)
    st.write('**Error (MSE)**')
    st.info(mean_squared_error(y_train, y_pred_train))

    st.subheader('**Test set**')
    st.write(f'**$R^2$(coefficient of determination)**')
    st.info(r2_test)
    st.write('**Error (MSE)**')
    st.info(mean_squared_error(y_test, y_pred_test))

if rad == "Dataset (after cleaning)":
    st.write("""
    The dataset was cleaned and units from certain columns, like torque, engine_cc, etc. was removed and only the numerical values were kept
    """)
    st.dataframe(df)
    st.header('Code')
    st.text("""
    Below is the code used to clean the original dataset, and get the above dataset
    """)
    code = """import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Car details v3.csv')
dataset.head()  
# counting missing values
dataset.isnull().sum()
# deleting the rows with missing values
dataset = dataset.dropna()
import re

# Creating new torque_rpm column with only numeric values

torque_list = dataset["torque"].to_list()
torque_rpm = []
def extraction_torque(x):
    for i in x:
        res = i.replace(",","")
        temp = [int(s) for s in re.findall(r'\d+', res)]
        torque_rpm.append(max(temp))
        
extraction_torque(torque_list)
print(torque_rpm[:2])

dataset['torque_rpm'] = torque_rpm
dataset.head(2)

# extracting mileage
mil_list = dataset['mileage'].to_list()
mil_kmpl = []
def extraction_mil(x):
for item in x:
    temp = []
    try:
    for s in item.split(" "):
        temp.append(float(s))
    except:
    pass
    mil_kmpl.append(max(temp))

extraction_mil(mil_list)
print(mil_list[:2])
print(mil_kmpl[:2])

dataset['mil_kmpl'] = mil_kmpl
dataset.head()

# for engine
engine_list = dataset['engine'].to_list()
engine_cc = []
def extraction_engine(x):
    for item in x:
        temp = []
        try:
            for s in item.split(" "):
                temp.append(float(s))
        except:
            pass
        engine_cc.append(max(temp))

extraction_engine(engine_list)
print(engine_list[:2])
print(engine_cc[:2])

dataset['engine_cc'] = engine_cc
dataset.head(2)

# for max power 
max_power_list = dataset['max_power'].to_list()
max_power_bhp = []

def extraction_maxpower(x):
    for item in x:
        temp = []
        try:
            for s in item.split(" "):
                temp.append(float(s))
        except:
            pass
        max_power_bhp.append(max(temp))
        
extraction_maxpower(max_power_list)
print(max_power_list[:2])
print(max_power_bhp[:2])

dataset["max_power_bhp"] = max_power_bhp
dataset.head()

# so now let us create a new set dataframe with the original text columns deleted
data_new = dataset.drop(["mileage","engine","max_power","torque"], axis = 1)
data_new.head()


df = data_new.pop('selling_price')
data_new['selling_price'] = df

# exporting the cleaned dataset as csv
data_new.to_csv('data.csv', index=False)
    """
    st.code(code, language='python')

if rad == "Predictor":
    pass
