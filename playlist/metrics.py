from keras import metrics


def _top_k_accuracy(k):
    def _func(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k)

    return _func


def top_k_accuracy_func_list(k_list):
    return list(map(lambda k: _top_k_accuracy(k), k_list))
