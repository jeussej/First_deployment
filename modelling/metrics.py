#### listo ####

from sklearn.metrics import mean_absolute_error, make_scorer


def get_metric_name_mapping():
    return {_bike_number_error(): mean_absolute_error}


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _bike_number_error(): make_scorer(bike_number_error, greater_is_better=False, **params)
    }
    return mapping[name]


def _bike_number_error():
    return "bike_number_error"

def bike_number_error(y_true, y_pred, understock_price=0.3, overstock_price=0.7):
    import numpy as np
    error = (y_true - y_pred).astype(np.float32)
    factor = np.ones_like(error)
    factor[error > 0] = understock_price
    factor[error < 0] = overstock_price
    
    return np.sum(np.abs(error)*factor)/len(error)