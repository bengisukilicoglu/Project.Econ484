import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Veri yükle
data = pd.read_csv('dataset/imdb_top_1000.csv')
data.drop(['Poster_Link','Series_Title','Overview','Director','Star1','Star2','Star3','Star4'], axis=1, inplace=True)
data.rename(columns={
    'Released_Year':'Release Year',
    'Certificate':'Age Rating',
    'IMDB_Rating':'IMDB Rating',
    'Meta_score':'Metascore',
    'No_of_Votes':'Votes',
    'Gross':'Gross Revenue'
}, inplace=True)
data = data[data['Gross Revenue'].notna()]
data['Age Rating'] = data['Age Rating'].map({
    'U':'U','G':'U','PG':'U','GP':'U','TV-PG':'U',
    'UA':'UA','PG-13':'UA','U/A':'UA','Passed':'UA','Approved':'UA',
    'A':'A','R':'A'
})
data = data[data['Age Rating'].notna()]
data['Metascore Exists'] = data['Metascore'].notnull()
data.drop('Metascore', axis=1, inplace=True)
data = data[data['Release Year'].str.match(r'\d\d\d\d')]
data['Release Year'] = data['Release Year'].astype(int)
data['Runtime'] = data['Runtime'].str[:-4].astype(int)
data['Gross Revenue'] = data['Gross Revenue'].str.replace(',', '').astype(int) * (10**-6)
data['Genres'] = data['Genre'].apply(lambda x: len(x.split(', ')))
data['Primary Genre'] = data['Genre'].str.split(', ').str[0]
data.drop('Genre', axis=1, inplace=True)
data['Age Rating'] = data['Age Rating'].map({'U': 0, 'UA': 1, 'A': 2})
data = pd.get_dummies(data, columns=['Primary Genre'])

# Modeli yükle
model = joblib.load('rf_model.pkl')

# X, y ayır
X = data.drop('Gross Revenue', axis=1)
y = data['Gross Revenue']

# Tahmin yap
y_pred = model.predict(X)

# Karşılaştırma grafiği
plt.scatter(y, y_pred, alpha=0.7, color='purple')
plt.title("Gerçek vs Tahmin Edilen Gross Revenue")
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin Edilen Değer")
plt.grid(True)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')  # y = x çizgisi
plt.show()
