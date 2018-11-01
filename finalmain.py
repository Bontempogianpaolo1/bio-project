import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from Clustering import plot_clusters
from Cleaner import clean
from Plot import plot_diagram, plot_roc, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from Preprocessing.FeatureLca import FeatureLca
from Preprocessing.FeaturePca import FeaturePca
from Preprocessing.FeatureSelection import FeatureSelection
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
try:
    import seaborn as sns
    # sns.set()
except :
    print("Warning: seaborn not found, some of the plots might no be displayed")
seed = 241819
np.random.seed(seed)
print("select file name")
print("1 for ./Data/table.csv")
print("2 for ./Data/example2.csv(first 100)")
action = input('')
if action == '1':
    file = "./Data/table.csv"
else:
    file = "./Data/example2.csv"
print("selected " + file)
print("=========================================================")
print("Analysis on breast cancer data.")
print("=========================================================")
print()
print("Please wait while data are loaded...")
df = pd.read_csv(file, index_col=0)
print("clean data from empty rows or nan")
df = clean(df)
print("Data successfully loaded.")
print()

while True:
    print("Please enter: ")
    print("- '1' to see some statistic about data and to remove a specified class")
    print("- '2' for data classificatition")
    print("- '3' for data clustering")
    print("- '4' to exit")

    action = input('')
    if action == '1':
        counts = df.iloc[:, 0].value_counts()
        names = counts.index
        x = np.arange(counts.shape[0])
        plot_diagram(df)
        print("Do you want to remove a specific class?(y/n)")
        action = input('')
        if action == 'y':
            for i, name in zip(x, names):
                print("press {:d}".format(i) + " for " + name)
            action = eval(input(''))
            name = names[action]
            df = df[df.l != name]
    elif action == "2":
        print()
        print()
        print("==================CLASSIFICATION==================")
        print()
        counts = df.iloc[:, 0].value_counts()
        names = counts.index
        n_labels = np.arange(counts.shape[0])
        y = df.get('l').values
        x = df.drop(['l', 'labels'], axis=1)
        print()
        print()
        print("insert testing set dimension... 0.0-1.0")
        action = eval(input(''))
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=action, random_state=seed)
        print()
        print()
        print("------------------ Data standardization ------------------")
        print("Standardizing data...")
        scaler = StandardScaler(with_std=True)
        scaler.fit(X_train)# fit only on training set to avoid influence of  test set's leaking information on model
        pd.options.mode.chained_assignment = None  # default='warn'
        X_train.loc[:, X_train.columns] = scaler.transform(X_train.loc[:, X_train.columns])
        X_test.loc[:, X_test.columns] = scaler.transform(X_test.loc[:, X_test.columns])
        print("Training mean is now 0:")
        print(X_train.mean(axis=0))
        print("Training std is now 1:")
        print(X_train.std(axis=0))
        print("Test mean is:")
        print(X_test.mean(axis=0))
        print("Test std is:")
        print(X_test.std(axis=0))
        print()
        print()
        print("------------------ Dimensionality reduction ------------------")
        print("Choose a domensionality reduction method:")
        print("- '0' for no dimensional reduction.")
        print("- '1' for Principal Component Analysis")
        print("- '2' for Linear Discriminant Analysis")
        print("- '3' for Feature Selection")
        plt.close('all')

        while True:
            dim_reduction = input('')
            print()
            if dim_reduction == '0':
                print("You chose not to perform any dimension reduction")
                feature = None
                start = 0
                end = 0
                interval = 0
                break
            elif dim_reduction == '1':
                print("You chose to perform Principal Component Analysis as a dimensional reduction technique")
                feature = FeaturePca()
                print("insert the first n_features of the array")
                start = eval(input(''))
                print("insert the last n_features of the array")
                end = eval(input(''))
                print("insert the interval between each n_component")
                interval = eval(input(''))
                print("PCA fitted.")
                break
            elif dim_reduction == '2':
                print("You chose to perform Linear Discriminant Analysis as a dimensional reduction technique")
                feature = FeatureLca()
                print("insert the first n_features of the array")
                start = eval(input(''))
                print("insert the last n_features of the array")
                end = eval(input(''))
                print("insert the interval between each n_component")
                interval = eval(input(''))
                break
            elif dim_reduction == '3':
                print("You chose to perform Linear Discriminant Analysis as a dimensional reduction technique")
                feature = FeatureSelection()
                print("insert the first n_features of the array")
                start = eval(input(''))
                print("insert the last n_features of the array")
                end = eval(input(''))
                print("insert the interval between each n_component")
                interval = eval(input(''))
                break
            else:
                feature = None
                start = 0
                end = 0
                interval = 0
        print()
        print()
        y_test_bal = y_test
        while True:
            print("------------------ Class balancing ------------------")
            print("Choose a class balancing method:")
            print("- 'None' for no balancing.")
            print("- 'Up' for Oversampling (all classes will have the number of elements in the majority class)")
            print("- 'Under' for Undersampling(all classes will have the number of elements in the majority class)")
            print("- 'Mix' for a mixed approach ")
            balancing = input('')
            if balancing == "None" or balancing == 'none':
                print("You chose not to perform any class balancing")
                X_train_bal = X_train.values
                y_train_bal = pd.Series(y_train)
                X_test_bal = X_test
                y_test_bal = y_test
            elif balancing == "Up" or balancing == 'up' or balancing == 'UP':
                print("You chose to perform oversampling")
                X_train_bal, y_train_bal = RandomOverSampler(random_state=seed).fit_sample(X_train, y_train)
                major = names[0]
                y_train_bal = pd.Series(y_train_bal)
                X_test_bal = X_test
                y_test_bal = y_test
                print("Training data contain now " + str(X_train_bal.shape[0]) + "samples")
                print("The class division is the following:")
                print(y_train_bal.value_counts())
            elif balancing == "Under" or balancing == 'under':
                print("You chose to perform under-sampling")
                X_train_bal, y_train_bal = RandomUnderSampler(random_state=seed).fit_sample(X_train, y_train)
                major = names[0]
                y_train_bal = pd.Series(y_train_bal)
                X_test_bal = X_test
                y_test_bal = y_test
                print("Training data contain now " + str(X_train_bal.shape[0]) + "samples")
                print("The class division is the following:")
                print(y_train_bal.value_counts())
            elif balancing == "Mix" or balancing == 'mix' or balancing == 'MIX':
                print("You chose to use a mixed approach")
                try:
                    X_train_bal, y_train_bal = SMOTEENN(random_state=seed).fit_sample(X_train, y_train)
                except Exception as e:
                    print(e)
                    continue
                major = names[0]
                y_train_bal = pd.Series(y_train_bal)
                X_test_bal = X_test
                y_test_bal = y_test
                print("Training data contain now " + str(X_train_bal.shape[0]) + "samples")
                print("The class division is the following:")
                print(y_train_bal.value_counts())
            else:
                print("Your input was not understood, please try again")
                X_train_bal = X_trai
                y_train_bal = y_train
                X_test_bal = X_test
                y_test_bal = y_test
            print()
            break
        print("------------------ Classification algorithm ------------------")
        print()
        cnames = ["random Forest", "DecisionTreeClassifier", "Neural network", "svm"]

        while True:
            i = 0
            X_train_cla = X_train_bal
            y_train_cla = y_train_bal
            X_test_cla = X_test_bal
            y_test_cla = y_test_bal
            for name in cnames:
                i = i + 1
                print("press {:d}".format(i) + " " + name)
            i = i + 1
            print("press {:d}".format(i) + " to go back to the main menu")
            action = eval(input(''))
            plt.close('all')
            if action == i:
                break

            elif action == 1:
                print("random forest")
                print("1 for manual")
                print("2 for search the best minimum sample split given mss_min,mss_max,mss_step")
                action2 = eval(input(''))
                if action2 == 1:
                    # Nearest Neighbors
                    print("manual")
                    print("insert minimum sample split")
                    mss = eval(input(''))
                    print("insert criterion(gini,entropy)")
                    criterion = input('')
                    print("insert k_fold")
                    k = eval(input(''))
                    if feature is not None:
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            clf = RandomForestClassifier(criterion=criterion, min_samples_split=mss)
                            score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                'test_score']
                            print("n_feature %d,score %f" % (n_feature, score.mean()))
                            ada_arr.append(clf)
                            metrics.append(score.mean())
                            pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," \
                                "n_components %d using %s" % (cnames[action - 1],
                                                              mss,
                                                              criterion,
                                                              pca_arr[max_matrics_arg].n_components,
                                                              feature.name)
                        ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla)
                        y_pred = ada_arr[max_matrics_arg].predict(X_test_pca)
                        print(title)
                    else:
                        clf = RandomForestClassifier(criterion=criterion, min_samples_split=mss).fit(X_train_cla,
                                                                                                     y_train_cla)
                        y_pred = clf.predict(X_test_cla)
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," \
                                " without feature reduction" % (cnames[action - 1], mss, criterion)
                elif action2 == 2:
                    print("fitting classifier")
                    print("optimal")
                    print("insert mss_min")
                    msfmin = eval(input(''))
                    print("insert mss_max")
                    msfmax = eval(input(''))
                    print("insert mss_step")
                    msfstep = eval(input(''))
                    print("insert criterion(  gini,entropy)")
                    criterion = input('')
                    print("insert k_fold")
                    k = eval(input(''))
                    msfarray = np.arange(msfmin, msfmax, msfstep)
                    if feature is not None:
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            for mss in msfarray:
                                clf = RandomForestClassifier(criterion=criterion, min_samples_split=mss)
                                score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                    'test_score']
                                print("n_feature %d,minimum samples split %d score %f" % (n_feature, mss, score.mean()))
                                metrics.append(score.mean())
                                ada_arr.append(clf)
                                pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        print("best n_features %d" % pca_arr[max_matrics_arg].n_components)
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla).predict(X_test_pca)
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," \
                                "n_components %d using %s" % (cnames[action - 1],
                                                              ada_arr[max_matrics_arg].min_samples_split,
                                                              criterion,
                                                              pca_arr[max_matrics_arg].n_components,
                                                              feature.name)

                        print(title)
                    else:
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for mss in msfarray:
                            clf = RandomForestClassifier(criterion=criterion, min_samples_split=mss)
                            score = cross_validate(clf, X_train_cla, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                'test_score']
                            print("msf %d,score %f" % (mss, score.mean()))
                            metrics.append(score.mean())
                            ada_arr.append(clf)
                            # pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = X_test_cla
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," % (cnames[action - 1],
                                                                                            ada_arr[max_matrics_arg]
                                                                                            .min_samples_split,
                                                                                            criterion)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_cla, y_train_cla).predict(X_test_pca)

                        print(title)
                else:
                    continue
            elif action == 2:
                print("Decision Tree")
                # Nearest Neighbors
                print("1 for manual")
                print("2 for search the best minimum sample split given mss_min,mss_max,mss_step")
                action2 = eval(input(''))
                if action2 == 1:

                    print("fitting classifier")
                    print("optimal")
                    print("insert minimum sample split")
                    mss = eval(input(''))
                    print("insert criterion(gini,entropy)")
                    criterion = input('')
                    print("insert k_fold")
                    k = eval(input(''))
                    if feature is not None:
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=mss)
                            score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                'test_score']
                            print("n_feature %d,score %f" % (n_feature, score.mean()))
                            ada_arr.append(clf)
                            metrics.append(score.mean())
                            pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," \
                                "n_components %d using %s" % (cnames[action - 1],
                                                              mss,
                                                              criterion,
                                                              pca_arr[max_matrics_arg].n_components,
                                                              feature.name)
                        ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla)
                        y_pred = ada_arr[max_matrics_arg].predict(X_test_pca)
                        print(title)
                    else:
                        clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=mss)
                        clf.fit(X_train_cla, y_train_cla)
                        y_pred = clf.predict(X_test_cla)
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," \
                                " without feature reduction" % (cnames[action - 1], mss, criterion)
                elif action2 == 2:
                    print("fitting classifier")
                    print("optimal")
                    print("insert mss_min")
                    msfmin = eval(input(''))
                    print("insert mss_max")
                    msfmax = eval(input(''))
                    print("insert mss_step")
                    msfstep = eval(input(''))
                    print("insert criterion(gini,entropy)")
                    criterion = input('')
                    print("insert k_fold")
                    k = eval(input(''))
                    msfarray = np.arange(msfmin, msfmax, msfstep)
                    if feature is not None:
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            for mss in msfarray:
                                clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=mss)
                                score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                    'test_score']
                                print("n_feature %d,minimun sample split %dscore %f" % (n_feature, mss, score.mean()))
                                metrics.append(score.mean())
                                ada_arr.append(clf)
                                pca_arr.append(feature.get())

                        max_matrics_arg = int(np.argmax(metrics))
                        print("best n_features %d" % pca_arr[max_matrics_arg].n_components)
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla).predict(X_test_pca)
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," \
                                "n_feature %d using %s" % (cnames[action - 1],
                                                           ada_arr[max_matrics_arg].min_samples_split,
                                                           criterion,
                                                           pca_arr[max_matrics_arg].n_components,
                                                           feature.name)
                        print(title)
                    else:
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for mss in msfarray:
                            clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=mss)
                            score = cross_validate(clf, X_train_cla, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                    'test_score']
                            print("msf %d,score %f" % (mss, score.mean()))
                            metrics.append(score.mean())
                            ada_arr.append(clf)
                            # pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = X_test_cla
                        title = "algorithm: %s,minimum sample split: %d ,criterion: %s," % (cnames[action - 1],
                                                                                            ada_arr[max_matrics_arg]
                                                                                            .min_samples_split,
                                                                                            criterion)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_cla, y_train_cla).predict(X_test_pca)

                        print(title)
                else:
                    continue
                # Random Forest

            elif action == 3:
                print("Neural Network")
                print("1 for manual")
                print("2 for search the best alpha given alpha_min,alpha_max,alpha_step")
                action2 = eval(input(''))
                if action2 == 1:
                    print("manual")
                    print("insert alpha")
                    alpha = eval(input(''))
                    print("insert k_fold")
                    k = eval(input(''))
                    if feature is not None:
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            clf = MLPClassifier(alpha=alpha)
                            score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                'test_score']
                            print("n_feature %d,alpha %f score %f" % (n_feature, alpha, score.mean()))
                            ada_arr.append(clf)
                            metrics.append(score.mean())
                            pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        title = "algorithm: %s,alpha %d ," \
                                "n_feature %d using %s" % (cnames[action - 1],
                                                           ada_arr[max_matrics_arg].alpha,
                                                           pca_arr[max_matrics_arg].n_components,
                                                           feature.name)
                        ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla)
                        y_pred = ada_arr[max_matrics_arg].predict(X_test_pca)
                        print(title)
                    else:
                        clf = MLPClassifier(alpha=alpha)
                        clf.fit(X_train_cla, y_train_cla)
                        y_pred = clf.predict(X_test_cla)
                        title = "algorithm: %s,alpha: %f, without feature reduction" % (cnames[action - 1], alpha)
                elif action2 == 2:
                    print("fitting classifier")
                    print("optimal")
                    print("insert alfa_min")
                    alfa_min = eval(input(''))
                    print("insert alfa_max")
                    alfa_max = eval(input(''))
                    print("insert alfa_step")
                    alfa_step = eval(input(''))
                    print("insert k_fold")
                    k = eval(input(''))
                    alfa_arr = np.arange(alfa_min, alfa_max, alfa_step)
                    if feature is not None:
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            for alfa in alfa_arr:
                                clf = MLPClassifier(alpha=alfa)
                                score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                    'test_score']
                                print("n_component %d,alfa %f,score %f" % (n_feature, alfa, score.mean()))
                                metrics.append(score.mean())
                                ada_arr.append(clf)
                                pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        title = "algorithm: %s,alpha: %d ," \
                                "n_features %d using %s" % (cnames[action - 1],
                                                            ada_arr[max_matrics_arg].alpha,
                                                            pca_arr[max_matrics_arg].n_components,
                                                            feature.name)
                        ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla)
                        y_pred = ada_arr[max_matrics_arg].predict(X_test_pca)
                        print(title)
                    else:
                        ada_arr = list()
                        metrics = list()
                        for alfa in alfa_arr:
                            clf = MLPClassifier(alpha=alfa)
                            score = cross_validate(clf, X_train_cla, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                'test_score']
                            print("n_component %d,score %f" % (alfa, score.mean()))
                            metrics.append(score.mean())
                            ada_arr.append(clf)
                        max_matrics_arg2 = int(np.argmax(metrics))
                        X_test_pca = X_test_cla
                        title = "algorithm: %s,alfa: %d" % (cnames[action - 1], ada_arr[max_matrics_arg2].alpha)
                        ada_arr[max_matrics_arg2].fit(X_train_cla, y_train_cla)
                        y_pred = ada_arr[max_matrics_arg2].predict(X_test_pca)
                        print(title)
                else:
                    continue
                # Random Forest

            elif action == 4:
                print("svm")
                print("fitting classifier")
                if feature is not None:
                    print("1 for insert manually C with gamma='auto' and kernel= linear")
                    print(
                        "2 for search the best C and gamma given Cmin : Cstep: Cmax, gammamin: gammastep: gammamax and "
                        "kernel = RBF")
                    print("3 for search the best C and given Cmin : Cstep: Cmax and kernel = linear")
                    action2 = input('')
                    if action2 == '1':
                        print("manual")
                        print("insert C")
                        C = eval(input(''))
                        print("insert k_fold")
                        k = eval(input(''))
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            clf = svm.SVC(C=float(C), random_state=seed, kernel='linear')
                            score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                'test_score']
                            print("n_feature %d,C %fscore %f" % (n_feature, float(C), score.mean()))
                            ada_arr.append(clf)
                            metrics.append(score.mean())
                            pca_arr.append(feature.get())
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        title = "algorithm: %s,C: %d,n_components %d using %s" % (cnames[action - 1],
                                                                                  C,
                                                                                  pca_arr[max_matrics_arg].n_components,
                                                                                  feature.name)
                        ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla)
                        y_pred = ada_arr[max_matrics_arg].predict(X_test_pca)
                        print(title)
                    elif action2 == '2':
                        print("svm")
                        print("fitting classifier")
                        print("optimal")
                        print("insert Cmin")
                        Cmin = eval(input(''))
                        print("insert Cmax")
                        Cmax = eval(input(''))
                        print("insert Cstep")
                        Cstep = eval(input(''))
                        print("insert gammamin")
                        gammamin = eval(input(''))
                        print("insert gammamax")
                        gammamax = eval(input(''))
                        print("insert gammastep")
                        gammastep = eval(input(''))
                        print("insert k_fold")
                        k = eval(input(''))
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            for gamma in np.arange(gammamin, gammamax, gammastep):
                                for C in np.arange(Cmin, Cmax, Cstep):
                                    clf = svm.SVC(C=float(C), gamma=float(gamma), random_state=seed)
                                    score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                        'test_score']
                                    print("n_feature %d,score %f,C %f,gamma %f" % (n_feature, score.mean(), C, gamma))
                                    metrics.append(score.mean())
                                    pca_arr.append(feature.get())
                                    ada_arr.append(clf)
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla).predict(X_test_pca)
                        title = "algorithm: %s,C: %f,gamma: " \
                                "%f n_components %d using %s" % (cnames[action - 1],
                                                                 ada_arr[max_matrics_arg].C,
                                                                 ada_arr[max_matrics_arg].gamma,
                                                                 pca_arr[max_matrics_arg].n_components,
                                                                 feature.name)

                        print(title)
                    elif action2 == '3':
                        print("insert Cmin")
                        Cmin = eval(input(''))
                        print("insert Cmax")
                        Cmax = eval(input(''))
                        print("insert Cstep")
                        Cstep = eval(input(''))
                        print("insert k_fold")
                        k = eval(input(''))
                        n_components = np.arange(start, end, interval)
                        metrics = list()
                        pca_arr = list()
                        ada_arr = list()
                        for n_feature in n_components:
                            feature.setcomponents(n_feature)
                            feature.fit(X_train_cla, y_train_cla)
                            X_tr_red = feature.transform(X_train_cla)
                            X_ts_red = feature.transform(X_test_cla)
                            for C in np.arange(Cmin, Cmax, Cstep):
                                clf = svm.SVC(C=float(C), kernel='linear', random_state=seed)
                                score = cross_validate(clf, X_tr_red, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                    'test_score']
                                metrics.append(score.mean())
                                pca_arr.append(feature.get())
                                ada_arr.append(clf)
                        max_matrics_arg = int(np.argmax(metrics))
                        X_test_pca = pca_arr[max_matrics_arg].transform(X_test_cla)
                        X_train_pca = pca_arr[max_matrics_arg].transform(X_train_cla)
                        title = "algorithm: %s,C: %f, n_components %d using %s" % (cnames[action - 1],
                                                                                   ada_arr[max_matrics_arg].C,
                                                                                   pca_arr[
                                                                                       max_matrics_arg].n_components,
                                                                                   feature.name)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_pca, y_train_cla).predict(X_test_pca)
                        print(title)
                    else:
                        continue
                else:
                    print("svn without feature reduction or projection")
                    print("1 for insert manually C with gamma='auto' and kernel= linear")
                    print(
                        "2 for search the best C and gamma given Cmin : Cstep: Cmax, gammamin: gammastep: gammamax and "
                        "kernel = RBF")
                    print("3 for search the best C and given Cmin : Cstep: Cmax and kernel = linear")
                    action2 = input('')
                    if action2 == '1':
                        print("manual")
                        print("insert C")
                        C = eval(input(''))
                        clf = svm.SVC(C=float(C), random_state=seed, kernel='linear').fit(X_train_cla, y_train_cla)
                        title = "algorithm: %s,C: %d" % (cnames[action - 1], C)
                        y_pred = clf.predict(X_test_cla)
                        print(title)
                    elif action2 == '2':
                        print("insert Cmin")
                        Cmin = eval(input(''))
                        print("insert Cmax")
                        Cmax = eval(input(''))
                        print("insert Cstep")
                        Cstep = eval(input(''))
                        print("insert gammamin")
                        gammamin = eval(input(''))
                        print("insert gammamax")
                        gammamax = eval(input(''))
                        print("insert gammastep")
                        gammastep = eval(input(''))
                        print("insert k_fold")
                        k = eval(input(''))
                        metrics = list()
                        ada_arr = list()
                        for gamma in np.arange(gammamin, gammamax, gammastep):
                            for C in np.arange(Cmin, Cmax, Cstep):
                                clf = svm.SVC(C=float(C), gamma=float(gamma),
                                              random_state=seed)
                                score = cross_validate(clf, X_train_cla, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                    'test_score']
                                metrics.append(score.mean())
                                ada_arr.append(clf)
                        max_matrics_arg = int(np.argmax(metrics))
                        title = "algorithm: %s,C: %f,gamma: %f " % (cnames[action - 1],
                                                                    ada_arr[max_matrics_arg].C,
                                                                    ada_arr[max_matrics_arg].gamma)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_cla, y_train_cla).predict(X_test_cla)
                        print(title)
                    elif action2 == '3':
                        print("insert Cmin")
                        Cmin = eval(input(''))
                        print("insert Cmax")
                        Cmax = eval(input(''))
                        print("insert Cstep")
                        Cstep = eval(input(''))
                        print("insert k_fold")
                        k = eval(input(''))
                        metrics = list()
                        ada_arr = list()
                        for C in np.arange(Cmin, Cmax, Cstep):
                            clf = svm.SVC(C=float(C), kernel='linear', random_state=seed)
                            score = cross_validate(clf, X_train_cla, y_train_cla, cv=StratifiedKFold(n_splits=k))[
                                'test_score']
                            metrics.append(score.mean())
                            ada_arr.append(clf)
                        max_matrics_arg = int(np.argmax(metrics))
                        title = "algorithm: %s,C: %f," % (cnames[action - 1],  ada_arr[max_matrics_arg].C)
                        y_pred = ada_arr[max_matrics_arg].fit(X_train_cla, y_train_cla).predict(X_test_cla)
                        print(title)
                    else:
                        continue
            else:
                break

            print()
            print("Confusion matrix")
            totalscore = accuracy_score(y_test_cla, y_pred)
            print("final score : %f" % totalscore)
            cnf_matrix = confusion_matrix(y_test_bal, y_pred, labels=names)
            # plt.figure(figsize=(10, 10))
            # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
            print()
            np.set_printoptions(precision=2)
            # PlotDir non-normalized confusion matrix
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(cnf_matrix, classes=names,
                                  title=title)
            # PlotDir normalized confusion matrix
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(cnf_matrix, classes=names, normalize=True,
                                  title=title)
            print(classification_report(y_test_bal, y_pred, labels=names))
    elif action == "3":

        print()
        print()
        print("==================CLUSTERING==================")
        print()
        counts = df.iloc[:, 0].value_counts()
        names = counts.index
        n_labels = np.arange(counts.shape[0])
        y = df.get('l').values
        x = df.drop(['l', 'labels'], axis=1)
        print()
        print()
        print("------------------ Dimensionality reduction ------------------")
        print("Choose a domensionality reduction method:")
        print("- '0' for no dimensional reduction.")
        print("- '1' for Principal Component Analysis")
        print("- '2' for Linear Discriminant Analysis")
        print("- '3' for Feature Selection")
        plt.close('all')

        while True:
            dim_reduction = input('')
            print()
            if dim_reduction == '0':
                print("You chose not to perform any dimension reduction")
                feature = None
                break
            elif dim_reduction == '1':
                print("You chose to perform Principal Component Analysis as a dimensional reduction technique")
                feature = FeaturePca()
                print("PCA fitted.")
                break
            elif dim_reduction == '2':
                print("You chose to perform Linear Discriminant Analysis as a dimensional reduction technique")
                feature = FeatureLca()
                break
            elif dim_reduction == '3':
                print("You chose to perform Linear Discriminant Analysis as a dimensional reduction technique")
                feature = FeatureSelection()
                break
            else:
                feature = None
        print()
        print()
        plot_clusters(x, y, feature, counts)
    elif action == "4":
        break
