from sklearn.metrics import roc_auc_score
from pandas import crosstab
from numpy import log, sum
from scipy.stats import ks_2samp


class Metrics:
    @staticmethod
    def gini(df, x, y):
        gm = df.groupby(x)[y].mean()
        return 2 * roc_auc_score(y_true=df[y], y_score=df[x].map(gm)) - 1

    @staticmethod
    def woe_iv(df, x, y):
        return crosstab(df[x], df[y], normalize='columns') \
                .assign(woe=lambda _: log(_[1] / _[0])) \
                .assign(iv=lambda _: _['woe'] * (_[1] - _[0]))

    @staticmethod
    def ks(predicted, target):
        return ks_2samp(predicted, target)[0]


metrics = Metrics()
