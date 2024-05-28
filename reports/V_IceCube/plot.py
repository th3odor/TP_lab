import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import uncertainties as un
import mrmr
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

signal = pd.read_csv('data_local/signal_train.csv', sep=',')
background = pd.read_csv('data_local/background_train.csv', sep=',')


## preprocessing ##


df = pd.concat((signal, background))

for feature in df.columns.values: ## remove MC truths
    if "Weight" in feature:
        df.drop(columns=[feature], inplace=True)
    elif "MC" in feature:
        df.drop(columns=[feature], inplace=True)
    elif "Corsika" in feature: 
        df.drop(columns=[feature], inplace=True)
    elif "I3EventHeader" in feature:
        df.drop(columns=[feature], inplace=True)

df.replace({np.inf: np.nan, -np.inf: np.nan}, value=None, inplace=True) # inf to nan
for feauture in df.columns.values: 
    if df[feauture].isna().sum() > 0.1*len(df[feauture]): # drop columns with many nans
        df.drop(columns=[feauture], inplace=True)
df.dropna(axis=0, how="any", inplace=True) # drop any rows left with nans

label = df["label"]
df.drop(columns=["label"], inplace=True)
keys = df.columns.values

scaler = StandardScaler() # rescale the data
X = scaler.fit_transform(df)
df = pd.DataFrame(X, columns=keys)


## feature selection ##

feautures_class = mrmr.mrmr_classif(X=df, y=label, K=100) ## determine the best 100 features in order  

num_plots = 4
plt.figure(figsize=(12,3))
for i in range(num_plots):
    plt.subplot(1,num_plots,i+1)
    plt.hist(background[feautures_class[i]], histtype = "step", label = "background")
    plt.hist(signal[feautures_class[i]], histtype = "step", label = "signal")
    plt.xlabel(feautures_class[i])
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.legend()
plt.savefig("build/features.pdf")
plt.close()


## model training functions ##


def num_features(model, model_name):
    
    n_feat = [5,10,20,30,40,50,60,70,80,90,100]

    accuracy = []
    precision = []
    roc_auc = []

    print(f"{model_name}:")
    for n in n_feat:
        X = df[feautures_class[:n]]
        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size = 0.5, random_state = 69)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        accuracy.append( accuracy_score(y_test, y_predict) )
        precision.append( precision_score(y_test, y_predict) )
        roc_auc.append( roc_auc_score(y_test, y_predict) )
        print(f"Tried {n} features")

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.scatter(n_feat ,accuracy)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.subplot(1,3,2)
    plt.scatter(n_feat, precision)
    plt.xlabel("Number of features")
    plt.ylabel("Precision")
    plt.subplot(1,3,3)
    plt.scatter(n_feat, roc_auc)
    plt.xlabel("Number of features")
    plt.ylabel("ROC Score")
    plt.tight_layout()
    plt.savefig(f"build/{model_name}_features.pdf")
    plt.close()

def model_eval(model, model_name, n_feautures):
    X = df[feautures_class[:n_feautures]] ## take n feautures only
    kfold = KFold(n_splits=5, shuffle=True) ## kfolding with shuffling
    cv_result = cross_validate(model, X, label, scoring=["accuracy", "precision", "roc_auc"], cv = kfold, return_estimator=True)
    print("")
    print(f"{model_name} Cross validation result:")
    print(f"Accuracy: {un.ufloat(np.mean(cv_result['test_accuracy']),np.std(cv_result['test_accuracy']))}")
    print(f"Precision: {un.ufloat(np.mean(cv_result['test_precision']),np.std(cv_result['test_precision']))}")
    print(f"ROC: {un.ufloat(np.mean(cv_result['test_roc_auc']),np.std(cv_result['test_roc_auc']))}")
    print("")

    model_fitted = cv_result["estimator"][0] ## use one of the fitted models
    y_predict = model_fitted.predict(X)
    c_mat = confusion_matrix(label, y_predict)
    ax = sn.heatmap(c_mat, annot=True, cmap = "Blues", fmt='g',cbar_kws={'label': 'Counts'})
    ax.set(xlabel="Actual label", ylabel="Predicted label")
    plt.tight_layout()
    plt.savefig(f'build/{model_name}_confusion.pdf')
    plt.close()

    y_predict_proba = model_fitted.predict_proba(X)
    FP, TP, _ = roc_curve(label, y_predict_proba[:,1], pos_label=1)
    return model_fitted ,FP,TP


## Naive Bayes ##


num_features(GaussianNB(), "NaiveBayes")
NB, FP_NB, TP_NB = model_eval(GaussianNB(), "NaiveBayes", 5)


## Random forest ##


num_features(RandomForestClassifier(10, max_depth=10), "RandomForest")
RF, FP_RF, TP_RF = model_eval(RandomForestClassifier(10, max_depth=10), "RandomForest", 60)


## Neural Network ##


num_features(MLPClassifier(5), "NeuralNetwork")
NN, FP_NN, TP_NN = model_eval(MLPClassifier(5), "NeuralNetwork", 50)


## Comparing classifiers ##


plt.plot(FP_NB, TP_NB, label = "Naive Bayes")
plt.plot(FP_RF, TP_RF, label = "Random Forest")
plt.plot(FP_NN, TP_NN, label = "Neural Network")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.tight_layout()
plt.savefig("build/roc_curves.pdf")


## Predicting on test data ##


n=60
test = pd.read_csv('data_local/test.csv', sep=',')
X = test[keys]
X = scaler.transform(X)
X = pd.DataFrame(X, columns=keys)
X = X[feautures_class[:n]]
X.replace(to_replace=np.nan, value=0, inplace=True) # stoopid nan
labels_predicted = RF.predict(X) # Random forest performs the best so lets use it
print("")
print(f"Found {np.sum(labels_predicted)} signal events.")
print(f"Remaining {len(labels_predicted)-np.sum(labels_predicted)} events are background.")
print("")
print("")
result = pd.DataFrame(labels_predicted, columns=["label"])
result.to_csv("build/result.csv")