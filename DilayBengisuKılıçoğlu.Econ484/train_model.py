import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Veri yükle
data = pd.read_csv('dataset/imdb_top_1000.csv')

# Kolon temizleme
data.drop(['Poster_Link','Series_Title','Overview','Director','Star1','Star2','Star3','Star4'], axis=1, inplace=True)
data.rename(columns={
    'Released_Year':'Release Year',
    'Certificate':'Age Rating',
    'IMDB_Rating':'IMDB Rating',
    'Meta_score':'Metascore',
    'No_of_Votes':'Votes',
    'Gross':'Gross Revenue'
}, inplace=True)

# Null temizleme
data = data[data['Gross Revenue'].notna()]
data['Age Rating'] = data['Age Rating'].map({
    'U':'U','G':'U','PG':'U','GP':'U','TV-PG':'U',
    'UA':'UA','PG-13':'UA','U/A':'UA','Passed':'UA','Approved':'UA',
    'A':'A','R':'A'
})
data = data[data['Age Rating'].notna()]
data['Metascore Exists'] = data['Metascore'].notnull()
data.drop('Metascore', axis=1, inplace=True)

# Tip dönüşümleri
data = data[data['Release Year'].str.match(r'\d\d\d\d')]
data['Release Year'] = data['Release Year'].astype(int)
data['Runtime'] = data['Runtime'].str[:-4].astype(int)
data['Gross Revenue'] = data['Gross Revenue'].str.replace(',', '').astype(int) * (10**-6)
data['Genres'] = data['Genre'].apply(lambda x: len(x.split(', ')))
data['Primary Genre'] = data['Genre'].str.split(', ').str[0]
data.drop('Genre', axis=1, inplace=True)

# Sayısallaştır
data['Age Rating'] = data['Age Rating'].map({'U': 0, 'UA': 1, 'A': 2})
data = pd.get_dummies(data, columns=['Primary Genre'])

# 🎨 Grafik 1: IMDB Rating vs Gross Revenue
plt.scatter(data['IMDB Rating'], data['Gross Revenue'])
plt.title("IMDB Rating vs Gross Revenue")
plt.xlabel("IMDB Rating")
plt.ylabel("Gross Revenue (M)")
plt.grid(True)
plt.show()

# 🎨 Grafik 2: Runtime vs Gross Revenue
plt.scatter(data['Runtime'], data['Gross Revenue'], color='green')
plt.title("Runtime vs Gross Revenue")
plt.xlabel("Runtime")
plt.ylabel("Gross Revenue (M)")
plt.grid(True)
plt.show()

# 🎨 Grafik 3: Age Rating dağılımı
data['Age Rating'].value_counts().plot(kind='bar')
plt.title("Age Rating Distribution")
plt.xlabel("Age Rating")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# 🎯 Modelleme
X = data.drop('Gross Revenue', axis=1)
y = data['Gross Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

model = RandomForestRegressor(n_estimators=100, random_state=101)
model.fit(X_train, y_train)

# Tahmin & metrikler
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("✅ Model başarıyla eğitildi ve kaydedildi!\n")
print(f"🎯 Training Score: {train_score:.2%}")
print(f"🎯 Testing Score : {test_score:.2%}")
print(f"📉 MSE           : {mse:.2f}")
print(f"📉 MAE           : {mae:.2f}")
print(f"📈 R²            : {r2:.2f}")

# Kaydet
joblib.dump(model, 'rf_model.pkl')
