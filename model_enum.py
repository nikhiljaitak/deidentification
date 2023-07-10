from enum import Enum

class ModelEnum(Enum):
    SAVED_MODEL_PATH = "Models/Ensemble_Torch_NER/"
    TOKENIZER_SAVED_MODEL_PATH = "Models/Ensemble_Torch_NER/Tokenizer/"
    HUGGING_FACE_MODEL="studio-ousia/luke-large-finetuned-conll-2003"
    HUGGING_FACE_TOKENIZER="studio-ousia/luke-large-finetuned-conll-2003"
    APP_DOCKER_SEPERATOR="app/"
