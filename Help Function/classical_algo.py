import time
import tracemalloc

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def classic(data_train, data_test, label_train, label_test):
    start = time.time()
    tracemalloc.start()

    clf_rbf = SVC(kernel='rbf', gamma='scale')
    clf_rbf.fit(data_train, label_train)
    label_pred = clf_rbf.predict(data_test)
    acc_rbf = accuracy_score(label_test, label_pred)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end = time.time()

    return acc_rbf, end-start, peak