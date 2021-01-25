import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#print(train.head())
#print(test.head())

train_and_test = [train, test]

#print(train_and_test)
#타이틀 나누기
for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')

#print(train['Title'].value_counts())

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Major', 'Col',
                                                 'Countess', 'Lady', 'Don',
                                                 'Sir','Dona' 'Jonkheer','Capt'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

#print(train['Title'].value_counts())

#print(train.loc[(train['Title'] == 'Other'), 'Age'].mean())

#각 타이틀 별 평균 나이
Other_age = int(train.loc[(train['Title'] == 'Other'), 'Age'].mean())
Mr_age = int(train.loc[(train['Title'] == 'Mr'), 'Age'].mean())
Miss_age = int(train.loc[(train['Title'] == 'Miss'), 'Age'].mean())
Mrs_age = int(train.loc[(train['Title'] == 'Mrs'), 'Age'].mean())
Master_age = int(train.loc[(train['Title'] == 'Master'), 'Age'].mean())


#타이틀 별 평균 나이로 결측치 채우기
a = train[train['Title'] == 'Mr'].copy()
a['Age'].fillna(Mr_age, inplace=True)
b = train[train['Title'] == 'Miss'].copy()
b['Age'].fillna(Miss_age, inplace=True)
c = train[train['Title'] == 'Mrs'].copy()
c['Age'].fillna(Mrs_age, inplace=True)
d = train[train['Title'] == 'Master'].copy()
d['Age'].fillna(Master_age, inplace=True)
e = train[train['Title'] == 'Other'].copy()
e['Age'].fillna(Other_age, inplace=True)

a=a.append(b)
a=a.append(c)
a=a.append(d)
a=a.append(e)

train = a.copy()

a = test[test['Title'] == 'Mr'].copy()
a['Age'].fillna(Mr_age, inplace=True)
b = test[test['Title'] == 'Miss'].copy()
b['Age'].fillna(Miss_age, inplace=True)
c = test[test['Title'] == 'Mrs'].copy()
c['Age'].fillna(Mrs_age, inplace=True)
d = test[test['Title'] == 'Master'].copy()
d['Age'].fillna(Master_age, inplace=True)
e = test[test['Title'] == 'Other'].copy()
e['Age'].fillna(Other_age, inplace=True)

a=a.append(b)
a=a.append(c)
a=a.append(d)
a=a.append(e)

test = a.copy()

train_and_test = [train, test]

for dataset in train_and_test:
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)

#print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())


for dataset in train_and_test:
    dataset.loc[dataset['Age'] <=16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map({0:'child', 1:'young', 2:'middle', 3:'subold'
                                         , 4:'old'}).astype(str)

for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(13.675)
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in train_and_test:
    dataset.loc[dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
    dataset.loc[dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in train_and_test:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset["Family"] = dataset['Family'].astype(int)

features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop, axis = 1)
train = train.drop([ 'AgeBand', 'PassengerId' ], axis=1)

train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis = 1)
test_data = test.drop("PassengerId", axis = 1).copy()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

train_data, train_label = shuffle(train_data, train_label, random_state = 5)


def tat(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accur : ", accuracy, "%")
    return prediction

log_pred = tat(LogisticRegression())
svm_pred = tat(SVC())
knn_pred = tat(KNeighborsClassifier(n_neighbors = 4))
rf_pred = tat(RandomForestClassifier(n_estimators=100))
nb_pred = tat(GaussianNB())


sub = pd.DataFrame({"PassengerId":test["PassengerId"],
                    "Survived" : rf_pred
                    })

sub.to_csv('submission.csv', index=False)


