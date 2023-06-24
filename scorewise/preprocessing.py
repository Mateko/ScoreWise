from dataclasses import dataclass, field
from pandas import DataFrame, concat, qcut, get_dummies
from sklearn.metrics import roc_auc_score
from scorewise.metrics import metrics


@dataclass
class Preprocessing:
    df: DataFrame
    default: str
    preprocessing_columns: list = field(default_factory=lambda: [])
    quants_to_check: list = field(default_factory=lambda: [2, 5, 8, 12, 14, 16, 18])
    cat_columns: list = field(default_factory=lambda: [])
    binning_results: dict = field(default_factory=lambda: {})
    bins: dict = field(default_factory=lambda: {})
    discrete_col_names: list = field(default_factory=lambda: [])

    def check_variable_types(self):
        for col, col_type in zip(self.df.columns, self.df.dtypes):
            if col_type == 'object':
                self.cat_columns.append(col)
                continue

            if col_type != 'object' and col not in self.preprocessing_columns and col != self.default:
                self.preprocessing_columns.append(col)

    def drop_null_values(self):
        self.df.dropna(inplace=True)

        return self.df

    def null_values_to_mean(self):
        self.df.fillna(self.df.mean(), inplace=True)

        return self.df

    def extract_bin(self, column, quant):
        return concat([qcut(self.df[column], quant, duplicates='drop'), self.df[self.default]], axis=1)

    def test_quants(self):
        if len(self.preprocessing_columns) < 1:
            raise ValueError('Firstly check column types')

        for column in self.preprocessing_columns:
            for i, quant in enumerate(self.quants_to_check):
                _df = self.extract_bin(column, quant)
                if i == 0:
                    self.binning_results[column] = {
                        'no_bins': [quant],
                        'gini': [metrics.gini(_df, column, self.default)],
                        'iv': [metrics.woe_iv(_df, column, self.default)['iv'].sum()],
                        'bins': [bin_ for bin_ in _df[column].unique()]
                    }
                else:
                    self.binning_results[column]['no_bins'].append(quant)
                    self.binning_results[column]['gini'].append(metrics.gini(_df, column, self.default))
                    self.binning_results[column]['iv'].append(metrics.woe_iv(_df, column, self.default)['iv'].sum())
                    self.binning_results[column]['bins'].append([x for x in _df[column].unique()])

        return self.binning_results

    def pick_best_number_of_bins(self, metric):
        if metric not in ['gini', 'iv']:
            raise NotImplementedError('Pick one of: gini, iv. Other metrics not supported atm')

        if len(self.binning_results) < 1:
            raise ValueError('Firstly bin data')

        for col in self.binning_results.keys():
            index = self.binning_results[col][metric].index(max(self.binning_results[col][metric]))
            self.bins[col] = {
                metric: self.binning_results[col][metric][index],
                'no_bins': self.binning_results[col]['no_bins'][index],
                'bins': self.binning_results[col]['bins'][index]
            }

            print(col, metric, self.bins[col][metric], 'no_of_bins: ', self.bins[col]['no_bins'])

    def bin_data(self):
        if len(self.bins) < 1:
            raise ValueError('Firstly use test_quants method, or define bins at your own')

        for col in self.bins.keys():
            try:
                self.df[f'{col}_binned'] = qcut(self.df[col], self.bins[col]['no_bins'], duplicates='drop')
            except KeyError:
                raise ValueError('Trying to bin not specified column in bin dict.')

        return self.df

    def get_binned_cols(self):
        binned_cols = [c for c in self.df.columns if c.split('_')[-1] == 'binned']
        if len(binned_cols) > 0:
            return binned_cols
        else:
            raise ValueError('There is no binned data')

    def create_dummies_cols(self, dummies_for_na=False):
        _ = []
        for col in self.get_binned_cols():
            _.append(get_dummies(self.df[col], prefix=f'{col}', dtype='int', dummy_na=dummies_for_na))

        return concat(_, axis=1)

    def create_discrete_cols(self):
        _ = []
        for col in self.get_binned_cols():
            transformed = self.df[col].cat.codes
            new_col_name = f'{col}_[{transformed.min()},{transformed.max()}'
            self.discrete_col_names.append(new_col_name)

            _.append(DataFrame(transformed, columns=[new_col_name]))

        return concat(_, axis=1)
