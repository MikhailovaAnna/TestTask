import csv
import pandas as pd
import numpy as np
import datetime
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# чтение из файла 
def read_csv(filename1, filename2):
  df_users = pd.read_csv(filename1, sep=',')
  df_results = pd.read_csv(filename2, sep=',', engine='python')
  df_users['create_time'] = pd.to_datetime(df_users['create_time']) # перевод дат и времени в удобный формат
  df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
  return df_users, df_results


#запись в файл
def write_csv(filename, data, columns):
  df = pd.DataFrame(data, columns=columns)
  df.to_csv(filename, index=False)
  return df


# выбор тех фич, которые подходят под наши условия
def select_features(df_users, df_results):
  user_features = []
  for id in df_users['id']:
    id_index = df_users[df_users['id'] == id].index
    index_list = df_results[df_results['user_id'] == id].index
    prob_win = 0 # вероятность побед
    av_len = 0 # средняя длина игры
    av_magic = 0 # среднее количество магии
    
    for index in index_list:
      diff = df_results['timestamp'][index] - df_users['create_time'][id_index[0]]
      diff /= np.timedelta64(1, 'h') # вычислили разницу между началом регистрации и действием
      if diff < 24: # если с момента регистрации прошло меньше 24 часов, то считаем 
        prob_win += df_results['winner'][index]
        av_len += df_results['length'][index]
        av_magic += df_results['magic_used'][index]

    if len(index_list) > 0:
      note = [df_results['user_id'][index], prob_win/len(index_list), av_len/len(index_list), av_magic/len(index_list)]
      user_features.append(note) 
  return user_features


def heatmap(df):
  plt.figure(figsize=(6,4))
  sns.heatmap(df.corr(),cmap='Blues',annot=False)
  plt.show()


def corr_matrix(df, k):
  cols = df.corr().nlargest(k, 'user_id')['user_id'].index
  cm = df[cols].corr()
  plt.figure(figsize=(10,6))
  sns.heatmap(cm, annot=True, cmap = 'viridis')
  plt.show()


def distribution_plot(df, k):
  l = df.columns.values
  number_of_rows = len(l)-1/k
  plt.figure(figsize=(2*k, 5*number_of_rows))
  for i in range(0, len(l)):
      plt.subplot(number_of_rows + 1, k, i + 1)
      sns.distplot(df[l[i]],kde=True)
  plt.show()


# разделение данных для обучения и прогноза
def features_labels_prep(df_prep, df):
  features = []
  labels = []
  features_to_predict = [] # данные для прогноза
  for id in df_pred['id']: # для всех игроков, у кого есть метки 
    id_index = df[df['user_id'] == id].index
    feature = [df['prob_win'][id_index[0]], df['av_length'][id_index[0]], df['av_magic'][id_index[0]]]
    features.append(feature) # записываем его характеристики 
    labels.append(df_pred['prediction'][df_pred.loc[df_pred['id'] == id].index[0]]) # и метку

  id_to_predict = [] # те, чью характеристику необходимо прогнозировать
  id_list = df_pred.id.value_counts()
  for id in df['user_id']:
    if id not in id_list:
      id_index = df[df['user_id'] == id].index
      feature = [df['prob_win'][id_index[0]], df['av_length'][id_index[0]], df['av_magic'][id_index[0]]]
      features_to_predict.append(feature)
      id_to_predict.append(id)
  return features, labels, id_to_predict, features_to_predict


def SVC_model(X_train, X_test, y_train, y_test):
  SVC_model = SVC()
  SVC_model.fit(X_train, y_train)
  SVC_prediction = SVC_model.predict(X_test)
  print("Accuracy of SVC model: ", accuracy_score(SVC_prediction, y_test))  
  err_test  = np.mean(y_test  != SVC_prediction)
  print("Error test: ", err_test)
  # print("Confusion matrix of SVC model: ", confusion_matrix(SVC_prediction, y_test))
  return SVC_model


def logistic_regression(X_train, X_test, y_train, y_test):
  logistic_model = LogisticRegression()
  logistic_model.fit(X_train, y_train)
  predicted = logistic_model.predict(X_test)
  print("Accuracy of logistic regression: ", accuracy_score(predicted, y_test))
  err_test  = np.mean(y_test  != predicted) 
  print("Error test: ", err_test)
  # print(metrics.confusion_matrix(predicted, y_test))
  return logistic_model


def NB_model(X_train, X_test, y_train, y_test):
  NBmodel = GaussianNB()
  NBmodel.fit(X_train, y_train)
  predicted = NBmodel.predict(X_test)
  print("Accuracy of NB model: ", accuracy_score(predicted, y_test)) 
  err_test  = np.mean(y_test  != predicted) 
  print("Error test: ", err_test)
  # print(metrics.confusion_matrix(predicted, y_test))
  return NBmodel


def result_form(id_to_predict, predicted_labels):
  predicted_results = []
  for i in range(len(predicted_labels)):
    note = [id_to_predict[i], predicted_labels[i]]
    predicted_results.append(note)
  return predicted_results


if __name__ == "__main__":
  # задание 1
  filename1 = 'users.csv'
  filename2 = 'game_results.csv'
  df_users, df_results = read_csv(filename1, filename2)

  user_features = select_features(df_users, df_results)
  filename = 'user_features.csv'
  columns = ["user_id", "prob_win", "av_length", "av_magic"]
  df = write_csv(filename, user_features, columns)
  heatmap(df)
  k = 4 
  corr_matrix(df, k)
  distribution_plot(df, k)

  # задание 2
  df_pred = pd.read_csv('predictions.csv', sep=',')
  features, labels, id_to_predict, features_to_predict = features_labels_prep(df_pred, df)
  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=32)
  SVC_model = SVC_model(X_train, X_test, y_train, y_test)
  LogisticRegression = logistic_regression(X_train, X_test, y_train, y_test)
  NB_model = NB_model(X_train, X_test, y_train, y_test)

  predicted_labels = SVC_model.predict(features_to_predict)
  predicted_results = result_form(id_to_predict, predicted_labels)
  result_filename = 'user_prediction.csv'
  df_res = write_csv(result_filename, predicted_results, ["user_id", "prediction"])
