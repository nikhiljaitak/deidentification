
import inspect
import sys
import re
import nltk
import time
#nltk.download('stopwords')
from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import LukeTokenizer, LukeForEntitySpanClassification
from transformers import LukeConfig, LukeModel

import logging

# Configure logging at application startup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
# Define a rotating file handler for logging to a file
file_handler = logging.FileHandler(filename='app.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))

# Add the file handler to the root logger
logging.getLogger().addHandler(file_handler)

class NamedEntityRecognition:

    SIMILARITY_THRESHOLD=0.65
    BLANK_STRING=''
    REGEX_ONLY_DIGITS='[^0-9]'
    REGEX_MEMBER_ID=['\s?[wW]\d{9}\s?', 'w(\d+){9}\s?','w(\d+){4}\s?(\d){5}\s?','w(\d+){5}\s(\d){4}\s?','w\s?(\d+){9}\s?','mbr#*\s*(\d+){7}\s*','mbr*\s*(\d+){7}\s*','-\s*(\d+){7}\s*','\s*(\d+){7}\s*-','member id=([0-9]){9}\s?','\s\d{7}\s','\s\d{8}\s','\s\d{9}\s','\s\d{10}\s']

    def clean_text(self, text):
        print('before total words=',len(text.split()))
        from string import punctuation
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in tokens if word not in stop_words]
        text = ' '.join(words)+' '
        print('total words=',len(text.split()))
        return text
    
    
    def saveHuggingFaceModel(outputPath,model):
        print("The current function name is:", inspect.currentframe().f_code.co_name)
        model.save_pretrained(outputPath)

    
    def loadSavedModel(self, savedModelpath):
        print("The current function name is:", inspect.currentframe().f_code.co_name)
        configuration = LukeConfig()
        LESC = LukeForEntitySpanClassification(configuration)
        savedModel = LESC.from_pretrained(savedModelpath)
        return savedModel
    
   
    def loadTokenizer( self, savedTokenizerPath):
        print("The current function name is:", inspect.currentframe().f_code.co_name)
        tokenizerModel = LukeTokenizer.from_pretrained(savedTokenizerPath)
        return tokenizerModel

    def prepareInputForLuke(self, textInput):
            pass

    def get_NER(self, lukeTokenizer, lukeModel, textInput):
        try:
            persons=[]
            print("The current function name is:", inspect.currentframe().f_code.co_name)
            startTime=time.time()
            print('Name Prediction Heavy Model ::',textInput)
            word_start_positions=[]
            word_end_positions=[]
            start=0
            end=0

            if len(textInput.split(' '))>512:
                print('Name cannot be extracted due to length bigger than expected')
                return []
            for word in textInput.split(' '):
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
                    
            modelObj = NamedEntityRecognition()
            #lukeTokenizer=modelObj.loadTokenizer('/Ensemble_Torch_NER/Tokenizer/')
            #lukeTokenizer=modelObj.loadTokenizer('/Users/nikhiljaitak/Downloads/Suresh_Sir_/Codes/Dev/Code_27Sept/SO_AI-ML_L1L2EmailIntentPredictor-master/Models/Ensemble_Torch_NER/Tokenizer/')
            #lukeModel=modelObj.getLukeModel()
            #lukeModel=modelObj.loadSavedModel('Ensemble_Torch_NER/')
            #lukeModel=modelObj.loadSavedModel('/Users/nikhiljaitak/Downloads/Suresh_Sir_/Codes/Dev/Code_27Sept/SO_AI-ML_L1L2EmailIntentPredictor-master/Models/Ensemble_Torch_NER/')
            #model.saveHuggingFaceModel('/Users/nikhiljaitak/Downloads/Suresh_Sir_/Codes/Dev/Code_27Sept/SO_AI-ML_L1L2EmailIntentPredictor-master/Models/Ensemble_Torch_NER/',lukeModel)
            #model.saveHuggingFaceModel('Models/Ensemble_Torch_NER/Tokenizer/',lukeTokenizer)
            inputs = lukeTokenizer(textInput, entity_spans=entity_spans, return_tensors="pt")
            
            outputs = lukeModel(**inputs)
            logits = outputs.logits
            predicted_class_indices = logits.argmax(-1).squeeze().tolist()
            persons=[]
            for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
                if predicted_class_idx != 0:
                #print(textInput[span[0] : span[1]], model.config.id2label[predicted_class_idx])
                    print('Execution Time :::',time.time()-startTime)
                    name=''
                    if lukeModel.config.id2label[predicted_class_idx] =='PER':
                        fullNamePrediction = textInput[span[0] : span[1]]
                        print('FULL NAME PREDICTION:----',fullNamePrediction)
                        name = fullNamePrediction.split()
                        if len(fullNamePrediction.split()) == 2:
                            persons.append(name[0]+' '+name[1])
                        elif len(fullNamePrediction.split()) == 1:
                            persons.append(name[0]+'')
                        elif len(fullNamePrediction.split()) >=3:
                            persons.append(fullNamePrediction)
        
            for person in persons:
                if len(person.split('.'))>1:
                    persons.remove(person)
                    for name in person.split('.'):
                        persons.append(name)

            for person in persons:
                if len(person.split(' '))>1:
                    persons.remove(person)
                    for name in person.split(' '):
                        persons.append(name)
        except Exception as exception:
            logging.exception('An exception occurred: %s in function name: %s', str(exception),str(inspect.currentframe().f_code.co_name))
            print(inspect.currentframe().f_code.co_name,exception)
            return exception
        return persons
    
    def hide_phi(text, entities):
        print("The current function name is:", inspect.currentframe().f_code.co_name)
        print(entities,text)
        for i in entities:
            text = text.replace(i,"***")
        return text

    def get_email_address(self, description):
            print("The current function name is:", inspect.currentframe().f_code.co_name)
            emailRegex='[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+'
            import re
            print(description)
            match=re.findall(emailRegex,description)
            if len(match)>0:
                return match
            else:
                return []
        
    def get_member_id(self, emailBody):
            print("The current function name is:", inspect.currentframe().f_code.co_name)
            memberids=[]
            try:
                import re
                for regex in NamedEntityRecognition.REGEX_MEMBER_ID:
                    regexMatch=re.findall(regex, str(emailBody).lower())
                    print(regex, regexMatch)
                    for match in regexMatch:
                         memberids.append(re.sub(NamedEntityRecognition.REGEX_ONLY_DIGITS, '', match))
            except Exception as e:
                print('exception+'+str(e),memberids)
            
            finalmemberids = list(filter(lambda x: len(str(x)) > 6, memberids))
            print(finalmemberids)
            return finalmemberids
            
    def get_ssn(self, text):
            print("The current function name is:", inspect.currentframe().f_code.co_name)
            ssnRegex='\d{3}-\d{2}-\d{4}'
            import re
            match=re.findall(ssnRegex,text)
            if len(match)>0:
                return match
            else:
                return []
    def get_phone_numbers(self,text):
        print("The current function name is:", inspect.currentframe().f_code.co_name)
        #print("The current function name is:", self.my_method.__name__)
        import phonenumbers

        for match in phonenumbers.PhoneNumberMatcher(text, None):
            numbers=[]
            if phonenumbers.is_valid_number(match.number):
                 #print(match.number)
                  numbers.append('+'+str(match.number.country_code)+str(match.number.national_number))
                  #print(numbers)
            return numbers
        return []
    def get_dob(self, text):
        print(dir(self))
        #print("The current function name is:", self.my_method.__name__)
        dates=[]
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append(ent.text)
        return dates
        
    def get_named_entities(self, text):
        print("The current function name is:", inspect.currentframe().f_code.co_name)
        from transformers import AutoTokenizer, LukeForEntitySpanClassification

        
        modelObj = NamedEntityRecognition()
        tokenizer=modelObj.loadTokenizer('/Ensemble_Torch_NER/Tokenizer/')
        model=modelObj.loadSavedModel('/Ensemble_Torch_NER/')
        word_start_positions=[]
        word_end_positions=[]
        
        start=0
        end=0
        for word in text.split(' '):
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

        print(start_pos,end_pos)    

        inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_indices = logits.argmax(-1).squeeze().tolist()
        entities = []
        for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
            if predicted_class_idx != 0:
                entities.append(text[span[0] : span[1]])
            print(text[span[0] : span[1]], model.config.id2label[predicted_class_idx])
            
        print('Entities extracted ',entities)

    def get_dob_re(self, input_text):
        print("The current function name is:", inspect.currentframe().f_code.co_name)
        import re
        date_formats = [r"\s*\d{1,2}/\d{1,2}/\d{4}\s*", r"\s*\d{4}-\d{1,2}-\d{1,2}\s*", r"\s*\d{1,2}\.\d{1,2}\.\d{4}\s*", r"\s*[A-Z][a-z]{2,8} \d{1,2}, \d{4}\s*", r"\s*\d{1,2} [A-Z][a-z]{2,8} \d{4}\s*",r"\s*\d{4}/\d{1,2}/\d{1,2}\s*","\d{1,2}/\d{1,2}/\d{4}",r"\s*\d{1,2}/\d{1,2}/\d{2}\s*"]
        dates = []
    
        for date_format in date_formats:
            dates += re.findall(date_format, input_text)
    
        return dates
    
    def remove_digits(self, text):
            return ''.join(c for c in text if not c.isdigit())
    

    

    def replace_mask_value(self, text, replacevalue, toMask ):
        maskedText = text.replace(toMask,'['+replacevalue+']')
        return maskedText

    def deidentify_information( self, lukeTokenizer, lukemodel, input_text):

        entities = NamedEntityRecognition().get_entities(lukeTokenizer, lukemodel, input_text)
        deidentified_text=input_text.lower()

        for entity in entities:
            deidentified_text=deidentified_text.replace(entity,' xxxx ')
            #print(entity,"    ",deidentified_text)
        
        deidentified_text = NamedEntityRecognition().remove_digits(deidentified_text)
        
        return deidentified_text

    def remove_confidentiality_notice(self, text):
        pattern = r"CONFIDENTIALITY NOTICE.*?attachments"
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def remove_large_numbers(self, text):
        import re
        pattern = "\+?\d{8,}" # Matches digits with more than 7 digits
        replaced_text=text
        numbers_Regex = re.findall(pattern, replaced_text)
        print('NUMBERS REGEX::', numbers_Regex)
        for number in numbers_Regex:
            replaced_text=replaced_text.replace(number,'[NUMBER]')
        return replaced_text
    
    import re

    def remove_Numbers(self, input_text):
        pattern = "[0-9/()]"
        output_text = re.sub(pattern, "", input_text)

        return output_text

    
    def get_chatgpt_response(self, prompt):
        import requests
        print('prompt',prompt)
        api_endpoint = "https://api.openai.com/v1/engines/text-davinci-003/completions"
        api_key = "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"

        payload = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7,
         }

        headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        }

    
        response = requests.post(api_endpoint, json=payload, headers=headers)
        print('response_text',response.json())
    
        response_text = response.json()['choices'][0]['text']
    
        print(response_text)
        return response_text
    
    def get_entities(self, lukeTokenizer, lukemodel, input_text): 
        removed_Numbers_text=''
        try:
            entitities=[]
            EntityRecognitionObj = NamedEntityRecognition()
            print('before:',len(input_text.split()))
            input_text=EntityRecognitionObj.remove_confidentiality_notice(input_text)
            print('after:',len(input_text.split()))
            print("The current function name is:", inspect.currentframe().f_code.co_name)
            memberids=EntityRecognitionObj.get_member_id(input_text)
            emailaddresses= EntityRecognitionObj.get_email_address(input_text)
            ssns=EntityRecognitionObj.get_ssn(input_text)
            phonenumbers= EntityRecognitionObj.get_phone_numbers(input_text)
            dobs=EntityRecognitionObj.get_dob_re(input_text)
            cleanedText = EntityRecognitionObj.clean_text(input_text)
            print('CLEANED_TEXT----',cleanedText)
            names=EntityRecognitionObj.get_NER(lukeTokenizer, lukemodel,cleanedText)
            
            entitiesDictionary = {}
            entitiesDictionary['SSN']=ssns
            entitiesDictionary['MEMBERID']=memberids
            entitiesDictionary['EMAIL_ADDRESS']=emailaddresses
            
            entitiesDictionary['PHONENUMBER']=phonenumbers
            entitiesDictionary['DOB']=dobs
            entitiesDictionary['NAME']=names

            print(entitiesDictionary)
            entitities.append(ssns)
            entitities.append(memberids)
            entitities.append(emailaddresses)
            entitities.append(phonenumbers)
            entitities.append(dobs)
            entitities.append(names)

            # Iterate over the dictionary items
            for key, values in entitiesDictionary.items():

                print("Key:", key)
                print("Values:")
        
        # Iterate over the list of values for each key
                for value in values:
                    input_text = EntityRecognitionObj.replace_mask_value(input_text,key,value)
                    print(key,value," text now--",input_text)

            print("Person", names)
            print("Memberid",memberids)
            print("Email",emailaddresses)
            print("SSN",ssns)
            print("Phone number ",phonenumbers)
            print("DOB ",dobs)

            removed_Numbers_text=EntityRecognitionObj.remove_large_numbers(input_text)

            removed_Numbers_text=EntityRecognitionObj.remove_Numbers(input_text)

            allentities = [item for sublist in entitities for item in sublist]

            lowerallentities = [s.lower() for s in allentities]

            finalEntities = [entity for entity in lowerallentities if len(entity) >= 2]

            print("all entities:",finalEntities)
            print("------------")
            print("------------")
            print(" OUTPUT text:",removed_Numbers_text)
            print("------------")
            print("------------")
        except Exception as exception:
            logging.exception('An exception occurred: %s in function name: %s', str(exception),str(inspect.currentframe().f_code.co_name))
            print("Exception in ",inspect.currentframe().f_code.co_name,exception)
            return exception
        
        return removed_Numbers_text
    
input_text1 = "  ramesh kumar is living in california and have ssn as 123-32-2121 date is 11/08/2001 having memidd w123423112 and email  markj@aetna.com and phone num +12205344654 and this information has been sent by the mark jones"   
input_text2=  'Michael Davis born on 01/01/1950 has a member ID of w123423112, SSN 711-38-0829, and email michaeldavis@hotmail.com. Phone number is 12205344654'
input_text='''
**** External Email - Use Caution ****

My name is Kathrin Lemaich. I submitted to get my dental check up reward on 10/19. It never came. I called last Monday and they said they would look into it. I got an email Thursday 11/10 that said I would have it within 48 hours. That was up yesterday. We had no issues last year. 

Thank you. 
i am residing in California
kristine.white@alexlee.com
'''
input_text3='Patient John Smith, with the Social Security Number 123-45-6789, was born on January 15, 1980. You can contact him at (555) 123-4567. His address is 123 Main Street, Anytown, USA \r\n**** External Email - Use Caution ****\r\n\r\nMy name is Kathrin Lemaich. I submitted to get my dental check up reward on 10/19. It never came. I called last Monday and they said they would look into it. I got an email Thursday 11/10 that said I would have it within 48 hours. That was up yesterday. We had no issues last year. \r\n\r\nThank you. \r\ni am residing in California\r\nkristine.white@alexlee.com .\r\n**** External Email - Use Caution ****\r\n\r\nMy name is Kathrin Lemaich. I submitted to get my dental check up reward on 10/19. It never came. I called last Monday and they said they would look into it. I got an email Thursday 11/10 that said I would have it within 48 hours. That was up yesterday. We had no issues last year. \r\n\r\nThank you. \r\ni am residing in California\r\nkristine.white@alexlee.com . \r\n**** External Email - Use Caution ****\r\n\r\nMy name is Kathrin Lemaich. I submitted to get my dental check up reward on 10/19. It never came. I called last Monday and they said they would look into it. I got an email Thursday 11/10 that said I would have it within 48 hours. That was up yesterday. We had no issues last year. \r\n\r\nThank you. \r\ni am residing in California\r\nkristine.white@alexlee.com  .ramesh kumar is living in california and have ssn as 123-32-2121 date is 11/08/2001 having memidd w123423112 and email  markj@aetna.com and phone num +12205344654 and this information has been sent by the mark jones . ramesh kumar is living in california and have ssn as 123-32-2121 date is 11/08/2001 having memidd w123423112 and email  markj@aetna.com and phone num +12205344654 and this information has been sent by the mark jones '


print('**********')
#print(NamedEntityRecognition().get_member_id(input_text3))

print('****************************')
print('****************************')
#lukeTokenizer=NamedEntityRecognition().loadTokenizer(savedTokenizerPath='/Users/nikhiljaitak/Downloads/Suresh_Sir_/Codes/Dev/Code_27Sept/SO_AI-ML_L1L2EmailIntentPredictor-master/Models/Ensemble_Torch_NER/Tokenizer/')
#lukeModel=modelObj.getLukeModel()
#lukeModel=modelObj.loadSavedModel('Ensemble_Torch_NER/')
#lukeModel=NamedEntityRecognition().loadSavedModel('/Users/nikhiljaitak/Downloads/Suresh_Sir_/Codes/Dev/Code_27Sept/SO_AI-ML_L1L2EmailIntentPredictor-master/Models/Ensemble_Torch_NER/')
#print(NamedEntityRecognition().get_entities(lukeTokenizer, lukeModel, input_text2))
print('****************************')
print('****************************')
   