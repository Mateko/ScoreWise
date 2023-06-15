from dataclasses import dataclass
from pandas import read_sql, read_csv
from scorewise.config import settings


@dataclass
class LoadData:
    load_type: str
    custom_sql = None

    def __setattr__(self, prop, val):
        if prop == 'load_type':
            self._check_load_type(val)
            self._check_config(val)

        super().__setattr__(prop, val)

    @staticmethod
    def _check_load_type(val):
        if val not in ['csv', 'mysql']:
            raise NotImplementedError('Trying to use not implemented method, '
                                      'instead use csv or mysql')

    @staticmethod
    def _check_config(val):
        if val == 'csv':
            if len(settings.FILE_PATH) < 1 or len(settings.FILE_PATH) < 1:
                raise ValueError('Trying to load CSV data without setting FILE_PATH'
                                 'or FILE_NAME in config file')

        if val == 'mysql':
            if len(settings.DB_NAME) < 1 or len(settings.PASSWORD) < 1 or len(settings.USER) < 1 or \
                    len(settings.TABLE_NAME) < 1:
                raise ValueError('Trying to load MySql data without setting one of'
                                 'DB/PASSWORD/USER/DB')

    def load_data(self):
        if self.load_type == 'mysql':
            return read_sql(self.custom_sql if self.custom_sql else settings.TABLE_NAME, settings.DATABASE_URI)

        if self.load_type == 'csv':
            return read_csv(f'{settings.FILE_PATH}')
