import numpy as np
import pandas as pd
import seaborn as sb


#1. citanje podataka i prikaz prvih pet redova u tabeli
from sklearn.metrics import r2_score

from KnnAlgorithm import KnnAlgorithm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('datasets/cakes.csv')
print(data.head(), '\n')

#2. prikaz konciznih podataka o sadrzaju tabele i stat. inf. o svim atributima
print(data.info(), '\n')
print(data.describe(), '\n')
print(data.describe(include=[object]), '\n')

#3. eliminisanje primeraka sa nedostajucim vrednostima atributa ili popunjavanje

# provera
# print(data.loc[data['flour'].isnull()].head)
# print(data.loc[data['eggs'].isnull()].head)
# print(data.loc[data['sugar'].isnull()].head)
# print(data.loc[data['milk'].isnull()].head)
# print(data.loc[data['butter'].isnull()].head)
# print(data.loc[data['baking_powder'].isnull()].head)
# print(data.loc[data['type'].isnull()].head)

#svi imaju popunjene vrednosti

#4. graficki prikaz zavisnosti kontinualnih atributa koriscenjem korelacione matrice
ohe = OneHotEncoder(dtype=int, sparse_output=False)
fueltype = ohe.fit_transform(data.type.to_numpy().reshape(-1, 1))
# data.drop(columns=['type'], inplace=True)
data = data.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['type'])))

corr_matrix = data.select_dtypes(include=np.number).corr()
sb.heatmap(corr_matrix, annot=True, square=True, fmt='.1f')
plt.show()

#5. graficki prikaz zavisnosti izlaznog atributa od svakog ulaznog atributa rasejavajuci tacke
X = data.loc[:, ['flour']]
y = data['type']
sb.catplot(data=data, x="type", y="flour", kind="box")

X = data.loc[:, ['eggs']]
y = data['type']
sb.catplot(data=data, x="type", y="eggs", kind="box")

X = data.loc[:, ['sugar']]
y = data['type']
sb.catplot(data=data, x="type", y="sugar", kind="box")

X = data.loc[:, ['milk']]
y = data['type']
sb.catplot(data=data, x="type", y="milk", kind="box")

X = data.loc[:, ['butter']]
y = data['type']
sb.catplot(data=data, x="type", y="butter", kind="box")

X = data.loc[:, ['baking_powder']]
y = data['type']
sb.catplot(data=data, x="type", y="baking_powder", kind="box")
plt.show()

#6. graficki prikaz zavisnosti izlaznog atributa od svakog ulaznog kategorickog atributa koristeci odg tip grafika

#7. odabir atributa koji ucestvuju u treniranju modela
data_train = data.loc[:,['flour', 'eggs', 'sugar', 'milk','butter','baking_powder' ]] #DataFrame
labels = data.loc[:,'type'] #series

#8. izvrsavanje dodatnih transformacija nad odabranim atributima
data_train.eggs = data_train.eggs * 63

data_train.flour = (data_train.flour - data_train.flour.min())/(data_train.flour.max() - data_train.flour.min())
data_train.eggs = (data_train.eggs - data_train.eggs.min())/(data_train.eggs.max() - data_train.eggs.min())
data_train.sugar = (data_train.sugar - data_train.sugar.min())/(data_train.sugar.max() - data_train.sugar.min())
data_train.milk = (data_train.milk - data_train.milk.min())/(data_train.milk.max() - data_train.milk.min())
data_train.butter = (data_train.butter - data_train.butter.min())/(data_train.butter.max() - data_train.butter.min())
data_train.baking_powder = (data_train.baking_powder - data_train.baking_powder.min())/(data_train.baking_powder.max() - data_train.baking_powder.min())


#9. formiranje trening i test skupova podataka
X_train, X_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.7, random_state=123, shuffle=True)
# print(y_train)
#10. realizacija i testiranje modela

num_neighbors = int(np.sqrt(len(X_train)))
knn = KNeighborsClassifier(n_neighbors=num_neighbors)
knn.fit(X_train, y_train)
prediction = pd.Series(knn.predict(X_test))

print('Variance score: {}\n'.format(knn.score(X_test, y_test)))

my_knn = KnnAlgorithm()
my_knn.fit(X_train,y_train)
predicted = my_knn.predict(X_test,num_neighbors)
cnt_true = 0

for i in range(0, len(predicted)):
    if y_test.iloc[i] == predicted[i]:
        cnt_true = cnt_true + 1

res_df = pd.concat([y_test, pd.Series(data=predicted, name='Predicted', index=y_test.index)], axis=1)
print(res_df)

print("Preciznost mog modela: ", cnt_true/len(predicted))


