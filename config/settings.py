from pathlib import Path

"""
    MODEL_NAME:
        - cased:
            * It is important that difference between Onur and onur
        - multilingual:
            * I want it to detect languages other than English.
            * If you want to detect just English use bert-base-cased
    """
class Config:
    BASE_DIR=Path(__file__).parent.parent
    DATA_PATH="trspam.csv"
    DATA_DIR=BASE_DIR/"data"
    MODEL_DIR=BASE_DIR/"models"/"saved"
    MAX_LENGTH=128
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    EPOCHS=3