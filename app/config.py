from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    window_length : int = 30
    num_features : int = 21

    # These MUST match the columns used in the training dataframe
    sensor_columns: list = [f"sensor_{i}" for i in range(1, 22)]

    model_file_name: str = "model.pth"
    scaler_file_name: str = "scaler.pkl"

    class Config:
        env_file = ".env"

settings = Settings()