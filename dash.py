import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import plotly.express as px



inputFilePath="messy_data.csv"
inputFileCleanedPath="data.csv"

df=pd.read_csv(inputFilePath, na_values=[" "], skipinitialspace=True)

st.title("Projekt PAD")

st.subheader("Dane wejściowe")
st.dataframe(df)

df.columns = df.columns.str.replace(" ", "")
#df=df.replace(" ","")

df["clarity"]= df["clarity"].str.upper()
df["color"]= df["color"].str.upper()
df["cut"]= df["cut"].str.upper()


cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include="float64").columns.tolist()

imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(df[num_cols])
imp_mean.transform(df[num_cols])
df[num_cols]=imp_mean.transform(df[num_cols])

temp_df=df[cat_cols].replace("COLORLESS","C").replace( "IDEAL", "IDE.").replace( "PREMIUM", "PRE.").replace("VERY GOOD", "V.GOOD")

#enc = OrdinalEncoder()
#df[cat_cols]=enc.fit_transform(df[cat_cols])
#print( pd.Categorical(df["cut"], categories=[" FAIR", " GOOD", " VERY GOOD", " PREMIUM", " IDEAL"], ordered=True).codes)

df["cut"] = pd.Categorical(df["cut"], categories=["FAIR", "GOOD", "VERY GOOD", "PREMIUM", "IDEAL"], ordered=True).codes
df["color"] = pd.Categorical(df["color"], categories=["COLORLESS", "D", "E", "F", "G", "H","I","J"], ordered=True).codes
df["clarity"] = pd.Categorical(df["clarity"], categories=["IF","VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"], ordered=True).codes

st.subheader("Dane wejściowe po obróbce")
st.dataframe(df)


st.header("Wizualizacja danych")
st.subheader("Histogram atrybutów kategorycznych:")

count=0
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
#rename(columns={" COLORLESS": "C", " IDEAL": "IDE.", "PREMIUM": "PRE.","VERY GOOD": "V.GOOD"})
for i in temp_df:

    col = int((count%3))    
    sns.histplot(data=temp_df[[i]], x=i, ax=axs[col], legend=True, shrink=.8)
    sns.despine(fig, top=True, left=True, right=True)
    axs[col].tick_params(labelbottom=True)
    axs[col].set_title(i)        
    count=count+1

st.write()
st.pyplot(fig)

st.subheader("Violin plots with outliers:")

count=0
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 8), sharex=True)
for i in df.columns:
    data=df[i]
    row = int((count/5))
    col = int(count%5)
    q1, q3 = np.percentile(data, [25, 75])
    whisker_low = q1 - (q3 - q1) * 1.5
    whisker_high = q3 + (q3 - q1) * 1.5
    sns.violinplot(y=data.array, color="CornflowerBlue", ax=axs[row,col])
    outliers = data[(data > whisker_high) | (data < whisker_low)].array
    if(len(outliers)>0):
        sns.scatterplot(y=outliers, x=0, color="crimson", ax=axs[row,col])
        df=df[(df[i] < whisker_high) & (df[i] > whisker_low)]  
    sns.despine(fig, top=True, left=True, right=True)
    axs[row,col].set_title(i)        
    count=count+1
plt.setp(axs, "xticks", [])
plt.tight_layout()
st.write()
st.pyplot(fig)


st.subheader("Macierz korelacji atrybutów:")
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(df.corr(), annot=True, center=0.0, linewidths=.5, ax=ax)
st.write()
st.pyplot(fig)

st.subheader("Macierz korelacji atrybutów z skonsolidowanymi dimension:")
df["volume"]=df["xdimension"]*df["ydimension"]*df["zdimension"]
df=df.drop("xdimension", axis=1)
df=df.drop("ydimension",axis=1)
df=df.drop("zdimension",axis=1)
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(df.corr(), annot=True, center=0.0, linewidths=.5, ax=ax)
st.write()
st.pyplot(fig)

st.header("Regresja liniowa price(x)")
 
feature=st.selectbox("Wybierz zmienną zależną", ("carat","clarity","color","cut","volume","depth","table"))
if (st.button("Dopasuj model")):
    model = smf.ols(formula="price ~ {}".format(feature), data=df)
    model_fitted = model.fit()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.scatter(df[feature],df["price"], marker="o", color="blue", label="price vs {}".format(feature))
    ax.plot(df[feature], model_fitted.fittedvalues, color="red", label="Fitted regression line")
    plt.legend(loc="upper left")
    plt.xlabel("{}".format(feature))
    plt.ylabel("Price")
    st.write()
    st.pyplot(fig)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    line_y=[0]*len(df["price"])

    ax.scatter(df["price"],model_fitted.resid, marker="o", color="blue", label="Model residuals")
    ax.plot(df["price"], line_y, color="red", label="y=0")
    plt.legend(loc="upper left")
    plt.xlabel("Price")
    plt.ylabel("Residuals")
    st.write()
    st.pyplot(fig)

    st.write(model_fitted.summary(slim=True))


#print(len(df["price"]))

#print(len(lm_fit.fittedvalues))

#plot_scatter_and_line(df["price"],model.resid, )