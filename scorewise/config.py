class Settings:
    SERVER: str = "127.0.0.1:3306"
    USER: str = "test"
    PASSWORD: str = "test"
    DB_NAME: str = "test"
    TABLE_NAME: str = 'credit_risk_dataset'

    DATABASE_URI: str = (
        f"mysql+pymysql://{USER}:{PASSWORD}@{SERVER}/{DB_NAME}"
    )

    FILE_NAME = 'credit_risk_dataset.csv'
    FILE_PATH = f'data/{FILE_NAME}'


settings = Settings()
