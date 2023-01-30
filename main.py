import inline as inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

#odje ces
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from LinearRegressionGradientDescent import LinearRegressionGradientDescent

#1. ucitavanje skupa podataka i prikazivanje prvih pet redova iz tabele
data = pd.read_csv('datasets/fuels_consumptions.csv')
print(data.head(), '\n')

#2.prikaz inf o sadrzaju tabele i statistickih inf o svim atributima
print(data.info(), '\n')
print(data.describe(), '\n')
print(data.describe(include=[object]), '\n')


#3. eliminisanje primeraka sa nedostajucim vrednostima atributa ili popunjavanje

# provera
# print(data.loc[data['MODELYEAR'].isnull()].head)
# print(data.loc[data['MAKE'].isnull()].head)
# print(data.loc[data['MODEL'].isnull()].head)
# print(data.loc[data['VEHICLECLASS'].isnull()].head)

# print(data.loc[data['ENGINESIZE'].isnull()].head)
# ovde ima NaN vrednosti pa su popunjene sledecom linijom
data.ENGINESIZE = data.ENGINESIZE.fillna(data.ENGINESIZE.mean())

# print(data.loc[data['CYLINDERS'].isnull()].head)

# print(data.loc[data['TRANSMISSION'].isnull()].head)
# ovde ima Nan vrednosti koje su popunjene sa najcescom vrednoscu
data.TRANSMISSION = data.TRANSMISSION.fillna(data.TRANSMISSION.mode()[0])

# print(data.loc[data['FUELTYPE'].isnull()].head)
# ovde ima Nan vrednosti koje su popunjene sa najcescom vrednoscu
data.FUELTYPE = data.FUELTYPE.fillna(data.FUELTYPE.mode()[0])

# print(data.loc[data['FUELCONSUMPTION_CITY'].isnull()].head)
# print(data.loc[data['FUELCONSUMPTION_HWY'].isnull()].head)

# print(data.loc[data['FUELCONSUMPTION_COMB'].isnull()].head)
# print(data.loc[data['FUELCONSUMPTION_COMB_MPG'].isnull()].head)
# print(data.loc[data['CO2EMISSIONS'].isnull()].head)

#4. graficki prikaz zavisnosti kontinualnih atributa koriscenjem korelacione matrice
corr_matrix = data.select_dtypes(include=np.number).corr()
sb.heatmap(corr_matrix, annot=True, square=True, fmt='.1f')
# plt.show()

#5. graficki prikaz zavisnosti izlaznog atributa od svakog ulaznog kontinualnog
#atributa rasejavaci tacke po dekatrovom koordinatnom sistemu
X = data.loc[:, ['MODELYEAR']]
y = data['CO2EMISSIONS']
plt.figure('Dependency co2emissions from modelyear')
plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='modelyear')
plt.xlabel('Modelyear', fontsize=13)
plt.ylabel('Co2emissions', fontsize=13)
plt.title('Dependency co2emissions from modelyear')
plt.legend()
plt.tight_layout()
# plt.show()

X = data.loc[:, ['ENGINESIZE']]
plt.figure('Dependency co2emissions from enginesize')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='enginesize')
plt.xlabel('Enginesize', fontsize=13)
plt.ylabel('Co2emissions', fontsize=13)
plt.title('Dependency co2emissions from enginesize')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['CYLINDERS']]
plt.figure('Dependency co2emissions from cylinders')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='Cylinders')
plt.xlabel('Cylinders', fontsize=13)
plt.ylabel('Co2emissions', fontsize=13)
plt.title('Dependency co2emissions from cylinders')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_CITY']]
plt.figure('Dependency co2emissions from fuelconsumption_city')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='Fuelconsumption_city')
plt.xlabel('Fuelconsumption_city', fontsize=13)
plt.ylabel('Co2emissions', fontsize=13)
plt.title('Dependency co2emissions from fuelconsumption_city')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_HWY']]
plt.figure('Dependency co2emissions from fuelconsumption_hwy')
plt.scatter(X, y, s=23, c='blue', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='Fuelconsumption_hwy')
plt.xlabel('Fuelconsumption_hwy', fontsize=13)
plt.ylabel('Co2emissions', fontsize=13)
plt.title('Dependency co2emissions from fuelconsumption_hwy')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_COMB']]
plt.figure('Dependency co2emissions from fuelconsumption_comb')
plt.scatter(X, y, s=23, c='yellow', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='Fuelconsumption_comb')
plt.xlabel('Fuelconsumption_comb', fontsize=13)
plt.ylabel('Co2emissions', fontsize=13)
plt.title('Dependency co2emissions from fuelconsumption_comb')
plt.legend()
plt.tight_layout()

X = data.loc[:, ['FUELCONSUMPTION_COMB_MPG']]
plt.figure('Dependency co2emissions from fuelconsumption_comb_mpg')
plt.scatter(X, y, s=23, c='green', marker='o', alpha=0.7,
edgecolors='black', linewidths=2, label='Fuelconsumption_comb_mpg')
plt.xlabel('Fuelconsumption_comb_mpg', fontsize=13)
plt.ylabel('Co2emissions', fontsize=13)
plt.title('Dependency co2emissions from fuelconsumption_comb_mpg')
plt.legend()
plt.tight_layout()
# plt.show()

#6. graficki prikaz izlaznog atributa od svakog kategorickog koristeci odg tip grafika

plt.figure('Dependency co2emission from make')
sb.barplot(x='MAKE', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)

plt.figure('Dependency co2emission from model')
sb.barplot(x='MODEL', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)

plt.figure('Dependency co2emission from vehicle class')
sb.barplot(x='VEHICLECLASS', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)

plt.figure('Dependency co2emission from transmission')
sb.barplot(x='TRANSMISSION', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)

plt.figure('Dependency co2emission from fuel type')
sb.barplot(x='FUELTYPE', y='CO2EMISSIONS', data=data)
plt.xticks(rotation=90)
# plt.show()
# plt.show()

#7. odabir atributa koji ucestvuju u treniranju modela
data_train = data.loc[:,['ENGINESIZE', 'FUELCONSUMPTION_COMB', 'CYLINDERS', 'FUELTYPE']] #DataFrame
labels = data.loc[:,'CO2EMISSIONS'] #series
print(data_train.head(), '\n')

#8. izvrsavanje dodatnih transformacija nad odabranim atributima
ohe = OneHotEncoder(dtype=int, sparse_output=False)
fueltype = ohe.fit_transform(data_train.FUELTYPE.to_numpy().reshape(-1, 1))
data_train.drop(columns=['FUELTYPE'], inplace=True)
data_train = data_train.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['FUELTYPE'])))

# transmission = ohe.fit_transform(data_train.TRANSMISSION.to_numpy().reshape(-1, 1))
# data_train.drop(columns=['TRANSMISSION'], inplace=True)
# data_train = data_train.join(pd.DataFrame(data=transmission, columns=ohe.get_feature_names_out(['TRANSMISSION'])))

print(data_train.head(), '\n')

#9. formiranje trening i test skupova podataka
X_train, X_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.75, random_state=123, shuffle=False)

#10. realizacija i treniranje modela koristeci sve navedene pristupe
# spots = 200
# estates = pd.DataFrame(data=np.linspace(0, max(X_train['ENGINESIZE']), num=spots))

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print('Variance score: {}'.format(lr_model.score(X_test, y_test)), '\n')
plt.figure('Residual error')
plt.scatter(lr_model.predict(X_train), lr_model.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# graficko predstavljanje greske za test podatke
plt.scatter(lr_model.predict(X_test), lr_model.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

plt.legend(loc='upper right')
plt.title("Residual errors")

X_train_copy = X_train.copy(deep=True)
X_train.ENGINESIZE = (X_train.ENGINESIZE - X_train.ENGINESIZE.min())/(X_train.ENGINESIZE.max() - X_train.ENGINESIZE.min())
X_train.FUELCONSUMPTION_COMB = (X_train.FUELCONSUMPTION_COMB - X_train.FUELCONSUMPTION_COMB.min())/(X_train.FUELCONSUMPTION_COMB.max() - X_train.FUELCONSUMPTION_COMB.min())
X_train.CYLINDERS = (X_train.CYLINDERS - X_train.CYLINDERS.min())/(X_train.CYLINDERS.max() - X_train.CYLINDERS.min())

X_test.ENGINESIZE = (X_test.ENGINESIZE - X_train_copy.ENGINESIZE.min())/(X_train_copy.ENGINESIZE.max() - X_train_copy.ENGINESIZE.min())
X_test.FUELCONSUMPTION_COMB = (X_test.FUELCONSUMPTION_COMB - X_train_copy.FUELCONSUMPTION_COMB.min())/(X_train_copy.FUELCONSUMPTION_COMB.max() - X_train_copy.FUELCONSUMPTION_COMB.min())
X_test.CYLINDERS = (X_test.CYLINDERS - X_train_copy.CYLINDERS.min())/(X_train_copy.CYLINDERS.max() - X_train_copy.CYLINDERS.min())

# X_train.ENGINESIZE = X_train.ENGINESIZE/X_train.ENGINESIZE.max()
# X_train.FUELCONSUMPTION_COMB = X_train.FUELCONSUMPTION_COMB/X_train.FUELCONSUMPTION_COMB.max()
# X_train.CYLINDERS = X_train.CYLINDERS/X_train.CYLINDERS.max()

# X_test.ENGINESIZE = X_test.ENGINESIZE/X_train_copy.ENGINESIZE.max()
# X_test.FUELCONSUMPTION_COMB = X_test.FUELCONSUMPTION_COMB/X_train_copy.FUELCONSUMPTION_COMB.max()
# X_test.CYLINDERS = X_test.CYLINDERS/X_train_copy.CYLINDERS.max()
y_train_copy = y_train.copy(deep=True)
y_train = (y_train - y_train.min())/(y_train.max() - y_train.min())
# y_train = y_train / y_train.max()

print(X_train.head(), '\n')

lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train, y_train)
learning_rates = np.array([[0.4], [0.5], [0.3], [0.2], [0.5], [0.3], [0.5], [0.3]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates,1000)
plt.figure('mse')
plt.plot(np.arange(0, len(mse_history)), mse_history)

labels_predicted = lrgd.predict(X_test) * (y_train_copy.max() - y_train_copy.min()) + y_train_copy.min()
ser_predicted = pd.Series(data=labels_predicted, name='predicted', index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_predicted], axis=1)

print(res_df, '\n')
print('Preciznost mog modela: ', r2_score(np.array(y_test), labels_predicted), end='\n \n')


#11. prikaz dobijenih parametara modela, vrednosti fje greske i modela za sve realizovane pristupe

plt.show()

