from pandas import DataFrame
from dataclasses import dataclass, field
from numpy import argmax, arange, ones
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from scorewise.preprocessing import Preprocessing
from scorewise.metrics import metrics


@dataclass
class Selection:
    df: DataFrame
    default: str
    below_threshold: str = 'keep'
    diff_check: int = 10
    iv: float = 0.02
    gini: float = 0.05
    results: dict = field(default_factory=lambda: {})
    final_variables: list = field(default_factory=lambda: [])

    def __post_init__(self):
        self.pre = Preprocessing(self.df, default=self.default)
        self.pre.check_variable_types()

    def check_metrics(self, quant=10):
        for col in self.pre.preprocessing_columns:
            df_ = self.pre.extract_bin(col, quant)
            self.results[col] = {
                **metrics.gini_iv(df_, col, self.default)
            }

        if self.below_threshold == 'drop':
            self.df.drop(columns=[res for res, val
                                  in self.results.items()
                                  if float(val['gini']) < self.gini
                                  or float(val['iv']) < self.iv], inplace=True)

        self.final_variables = [col for col in self.df.columns if
                                col != self.default and col not in self.pre.cat_columns]

        return DataFrame(self.results)

