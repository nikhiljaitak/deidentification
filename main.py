from fastapi import FastAPI, Form
from pydantic import BaseModel, Field
from models import Model
import inspect
from controller_ import Controller
import json
from app_enum import AppEnum

app = FastAPI()

model, tokenizer = Model.load_model()

@app.get("/")
def root():
    return {"message": AppEnum.RUNNING_STATUS_MESSAGE.value}


class validate_request_body(BaseModel):
    input_text: str = Field(min_length=5, max_length=5000)
    task: str = Field(regex="^(deidentify_chatgpt|deidentify|deidentify_chatgpt_classification)$")
    uniqueid: str = Field(min_length=1)
    cgpt_endpoint: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    num_responses: int = Field(gt=0)
    temperature: float = Field(gt=0)
    regen_temperature: float = Field(gt=0)
    api_key: str = Field(min_length=1)
    history_conversations: str = Field(min_length=0, max_length=2000)
    
def get_output_dictionary(body):
    
    input_text = body.input_text
    task = body.task
    uniqueid = body.uniqueid
    cgpt_endpoint = body.cgpt_endpoint
    max_tokens = body.max_tokens
    num_responses = body.num_responses
    temperature = body.temperature
    regen_temperature = body.regen_temperature
    api_key = body.api_key
    history_conversations = body.history_conversations


    output_dict = {
        'input_text': input_text,
        'task': task,
        'uniqueid': uniqueid,
        'cgpt_endpoint': cgpt_endpoint,
        'max_tokens': max_tokens,
        'num_responses': num_responses,
        'temperature': temperature,
        'regen_temperature': regen_temperature,
        'api_key': api_key,
        'history_conversations': history_conversations
    }

    return output_dict


@app.post("/messages")
async def process_form(body: validate_request_body):
    print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)

    input_dictionary =get_output_dictionary(body)
    controllerobj=Controller()
    print("model",model)
    print("tokenizer:",tokenizer)
    deidentified_output= controllerobj.get_entities(tokenizer, model, input_dictionary['input_text'])
    return {"message": "Form values processed successfully","deidentified":deidentified_output}
