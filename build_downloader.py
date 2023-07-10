from transformers import LukeForEntitySpanClassification, LukeTokenizer
from model_enum import ModelEnum
from app_enum import AppEnum

model_name = ModelEnum.APP_DOCKER_SEPERATOR.value+ModelEnum.HUGGING_FACE_MODEL.value
model_save_path = ModelEnum.APP_DOCKER_SEPERATOR.value+ModelEnum.SAVED_MODEL_PATH
tokenizer_save_path = ModelEnum.APP_DOCKER_SEPERATOR.value+ModelEnum.SAVED_MODEL_PATH

model = LukeForEntitySpanClassification.from_pretrained(model_name)
tokenizer = LukeTokenizer.from_pretrained(model_name)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print("Model and tokenizer downloaded and saved successfully.")
