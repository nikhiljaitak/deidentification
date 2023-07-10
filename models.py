import os
import inspect
import re
import time
from transformers import LukeTokenizer, LukeForEntitySpanClassification
from transformers import LukeConfig, LukeModel
from model_enum import ModelEnum

class Model:
    saved_model_path = ModelEnum.APP_DOCKER_SEPERATOR.value+ModelEnum.SAVED_MODEL_PATH.value
    saved_tokenizer_path = ModelEnum.APP_DOCKER_SEPERATOR.value+ModelEnum.TOKENIZER_SAVED_MODEL_PATH.value
    access_model_path=saved_model_path
    access_tokenizer_path= saved_tokenizer_path

    def get_luke_model(self):
        print(inspect.currentframe().f_code.co_name)
        configuration = LukeConfig()
        LESC = LukeForEntitySpanClassification(configuration)
        luke_model = LESC.from_pretrained(ModelEnum.HUGGING_FACE_MODEL.value)
        return luke_model

    def get_luke_tokenizer(self):
        print(inspect.currentframe().f_code.co_name)
        luke_tokenizer = LukeTokenizer.from_pretrained(ModelEnum.HUGGING_FACE_TOKENIZER.value)
        return luke_tokenizer
    
    def save_model(self,output_path , model):
        print(inspect.currentframe().f_code.co_name,output_path)
        model.save_pretrained(output_path)
    
    def load_savedmodel(self, saved_model_path):
        print(inspect.currentframe().f_code.co_name, "loading model:",saved_model_path)
        configuration = LukeConfig()
        LESC = LukeForEntitySpanClassification(configuration)
        saved_model = LESC.from_pretrained(saved_model_path)
        print("Loaded NER Model")
        return saved_model
    
    def load_tokenizer( self, saved_tokenizer_path):
        print("The current function name is:", inspect.currentframe().f_code.co_name, "loading tokenizer:", saved_tokenizer_path)
        saved_tokenizer = LukeTokenizer.from_pretrained(saved_tokenizer_path)
        return saved_tokenizer
    
    def get_names(self, luke_tokenizer, luke_model, text_input):
        print(inspect.currentframe().f_code.co_name, text_input)
        try:
            persons=[]
            word_start_positions=[]
            word_end_positions=[]
            start=0
            
            for word in text_input.split(' '):
                word_start_positions.append(start)
                word_end_positions.append(len(word)+start)
                start=len(word)+start+1
            
            entity_spans = []

            for i, start_pos in enumerate(word_start_positions):
                count=0
                for end_pos in word_end_positions[i:]:
                    count+=1
                    if count==4:
                        break
                    entity_spans.append((start_pos, end_pos))
            
            inputs = luke_tokenizer(text_input, entity_spans=entity_spans, return_tensors="pt")
            
            outputs = luke_model(**inputs)
            logits = outputs.logits
            predicted_class_indices = logits.argmax(-1).squeeze().tolist()
            persons=[]
            for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
                if predicted_class_idx != 0:
                    if luke_model.config.id2label[predicted_class_idx] =='PER':
                        full_name_prediction = text_input[span[0] : span[1]]
                        persons.append(full_name_prediction)
                        print('FULL NAME PREDICTION:----',full_name_prediction)
            single_names = []
        
            for person in persons:
            
                parts = person.replace(".", "").split()

                filtered_parts = [part for part in parts if len(part) > 2]

                if filtered_parts:
                    single_names.extend(filtered_parts)
        
        except Exception as exception:
            print(inspect.currentframe().f_code.co_name,exception)
            return exception
        
        return list(set(single_names))
    


    @staticmethod
    def load_model():
        import os
        print(inspect.currentframe().f_code.co_name,Model.saved_tokenizer_path, Model.saved_model_path)
        model=None
        tokenizer=None
        modelobj=Model()
        if os.path.exists(Model.saved_model_path):
            model = modelobj.load_savedmodel(Model.saved_model_path)
            print('exists in path', Model.saved_model_path)
        else:
            model = modelobj.get_luke_model()
            modelobj.save_model(Model.saved_model_path, model)
            print('doesnt exists in path', Model.saved_model_path)
            
        if os.path.exists(Model.saved_tokenizer_path):
            print('exists in path', Model.saved_tokenizer_path)
            tokenizer = modelobj.load_tokenizer(Model.saved_tokenizer_path)

        else:
            tokenizer = modelobj.get_luke_tokenizer()
            modelobj.save_model(Model.saved_tokenizer_path, tokenizer)
            print('doesnt exists in path', Model.saved_tokenizer_path)

        print("model_loaded:",model!=None)
        print("tokenizer_loaded:",tokenizer!=None)
        return model, tokenizer
