import autosklearn.classification as s
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix


# creates an autoML classifier
def create_classifier(total_time=3600, init_config=25, ensemble=50, best_ensembles=50, mem_limit=3072,
                      resample='holdout', resample_args=0.67, include=None, exclude=None,
                      tmp_folder=None, metrics=None):

    clf = s.AutoSklearnClassifier(time_left_for_this_task=total_time,
                                  initial_configurations_via_metalearning=init_config,
                                  ensemble_size=ensemble,
                                  ensemble_nbest=best_ensembles,
                                  memory_limit=mem_limit,
                                  include=include,
                                  exclude=exclude,
                                  resampling_strategy=resample,
                                  resampling_strategy_arguments=resample_args,
                                  tmp_folder=tmp_folder,
                                  n_jobs=-1,
                                  scoring_functions=metrics)
    return clf


# splits the train data and trains the model afterwards
def train_model(clf, X, y,  path,split=.20,save=True, fit=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=7)
    if fit:
        clf.fit(X_train, y_train, dataset_name='lines')
    if save:
        persist_model(clf, path)

    return clf, X_test, y_test


# calculate the scores of the prediction. (accuracy, f1, recall, precision)
# y_pred: the prediction
# y_true: the groundtruth
# micro: whether to use the micro version or the macro version of the score
def calc_score(y_true, y_pred, micro='micro'):
    acc = accuracy_score(y_true, y_pred)
    pred = precision_score(y_true, y_pred, average=micro)
    rec = recall_score(y_true, y_pred, average=micro)
    f1 = f1_score(y_true, y_pred, average=micro)

    return acc, pred, rec, f1


# show the performance over time
# clf: the classifier
def performance_over_time(clf):
    poT = clf.performance_over_time_
    poT.plot(
        x='Timestamp',
        kind='line',
        legend=True,
        title='Auto-sklearn accuracy over time',
        grid=True,
    )
    plt.show()


# print the confusion matrix
def print_confusion_matrix(clf, X_test, y_pred):
    plot_confusion_matrix(clf, X_test, y_pred, normalize='true')
    plt.show()


# saves the trained model to avoid hour long training whenever the program is started
def persist_model(clf, filename):
    dump(clf, filename)


# loads the model
def load_model(filename):
    return load(filename)


# calcs the score of input table
def calc_scores(clf, y_test, y_pred, X_test):
    acc, pred, rec, f1 = calc_score(y_test, y_pred)
    print('printing the scores of the train process')
    print('Accuracy = %s, prediction = %s, recall = %s, f1 = %s' % (acc, pred, rec, f1))
    print_confusion_matrix(clf, X_test, y_pred)
