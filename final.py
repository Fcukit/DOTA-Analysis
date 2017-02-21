
import time
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score

import numpy as np
import pandas

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Считываем обучающую выборку
features = pandas.read_csv('D:/Anaconda2/IPN/features.csv', index_col='match_id')
#Загрузим тестовую выборку 
test = pandas.read_csv('D:/Anaconda2/IPN/features.csv', index_col='match_id')

#Ищем столбцы, имеющие пустые значения (NaN)
rows = features.shape[0]
columns = features.columns.values.tolist()
not_nan_list = features[columns].isnull().sum().sort_values() / rows

features = features.fillna(0)
test = test.fillna(0)

# Удаляем признаки, которые недоступны в первую 5-минутку
features.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'
                ], axis=1, inplace=True)
test.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'
                ], axis=1, inplace=True)

#Задаем кросс-валидацию
kf = KFold(len(features), n_folds=5, shuffle=True, random_state=42)
#Список оценок для каждой итерации
scores = []
#Размеры деревьев для обучения модели
tree_sizes = [30, 40, 50, 60]

#Формируем обучающую выборку
train_Y = features['radiant_win'].values

features = features.drop('radiant_win', 1)
test = test.drop('radiant_win', 1)
train_X = features
test_X = test

'''
#Оценивание модели градиентного бустинга

for tree_size in tree_sizes:
    print tree_size, ' trees evaluating ---'
    start_time = datetime.datetime.now()
    
    model = GradientBoostingClassifier(n_estimators=tree_size, random_state=42)
      
    score = cross_val_score(model, train_X, train_Y, cv=kf, scoring='roc_auc')
    pred = model.predict_proba(test_X)[:, 1]
    print '--- evaluating end'    
    print 'Time elapsed:', datetime.datetime.now() - start_time
    print score
    scores.append(score)
print scores    
'''

#Считаем количество уникальных персонажей в игре
heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
values = []
for heroe in heroes:
    values.append(pandas.unique(train_X[heroe].values.ravel()))
print np.max(values)    

#Удалим информацию о героях из обучающей выборки
train_X.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1, inplace=True)

scalator = StandardScaler()
train_X = scalator.fit_transform(train_X)

print 'logistic regression is started'
scores = []
#Список коэффициентов С для L2-регуляризации
c_params_range = range(-6, 6)
c_params = [10 ** i for i in c_params_range]

for c_param in c_params:
    #Оценим качество линейной модели
    lg_regr = LogisticRegression(penalty='l2', C=c_param, n_jobs=-1 )
    score = cross_val_score(lg_regr, train_X, train_Y, cv=kf, scoring='roc_auc')
    print 'c param is ', c_param
    print 'score is', np.mean(score)
    scores.append(np.mean(score))
max_index = scores.index(np.max(scores))
best_c_param = 10 ** c_params_range[max_index]
print 'the best C param is ', best_c_param
'''

'''
#Логистическая регрессия с мешком слов

#Считываем обучающую выборку
features_2 = pandas.read_csv('D:/Anaconda2/IPN/features.csv', index_col='match_id')
features_2 = features_2.fillna(0)
features_2 = features_2.drop('radiant_win', 1)
train_X_2 = features_2

# N — количество различных героев в выборке
N = np.max(values)  

X_pick = np.zeros((train_X_2.shape[0], N))

for i, match_id in enumerate(train_X_2.index):

    for p in xrange(5):
        X_pick[i, train_X_2.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, train_X_2.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
print X_pick.shape

pick_scalator = StandardScaler()
X_pick = pick_scalator.fit_transform(X_pick)
X_pick = pandas.DataFrame(X_pick, index=train_X_2.index)
print 'pick - ', X_pick.shape
print 'train - ', train_X_2.shape

train_X_2 = pandas.concat([train_X_2, X_pick], axis = 1)
print train_X_2.shape

print 'logistic regression is started'
scores = []

c_params = [10 ** i for i in c_params_range]

for c_param in c_params:
    #Оценим качество линейной модели
    lg_regr = LogisticRegression(penalty='l2', C=c_param, n_jobs=-1 )
    score = cross_val_score(lg_regr, train_X_2, train_Y, cv=kf, scoring='roc_auc')
    print 'c param is ', c_param
    print 'score is', np.mean(score)
    scores.append(np.mean(score))
max_index = scores.index(np.max(scores))
best_c_param = 10 ** c_params_range[max_index]
print 'the best C param is ', best_c_param
'''















