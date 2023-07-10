
import inspect
import sys
import re
import time
import logging
from data_cleaning import Cleaning
from helpers import Helpers
from entities import Entities
from models import Model
from app_enum import AppEnum
class Controller:

    
    def get_entities(self, luke_tokenizer, luke_model, input_text): 

        helpersobj=Helpers()
        cleaningobj=Cleaning()
        entityobj=Entities()
        modelobj=Model()
        removed_numbers=''
        
        try:
            entitities=[]
            names=[]
            
            print('before:',len(input_text.split()))
            
            input_text=cleaningobj.remove_confidentiality_notice(input_text)
            
            print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
            
            memberids=entityobj.get_member_id(input_text)
            emailaddresses= entityobj.get_email_address(input_text)
            ssns=entityobj.get_ssn(input_text)
            phonenumbers= entityobj.get_phone_numbers(input_text)
            dobs=entityobj.get_dob_re(input_text)
            cleaned_data = cleaningobj.clean_text(input_text)
            
            print('CLEANED_TEXT----',cleaned_data)
            
            split_conversations=helpersobj.split_conversations(cleaned_data)
            
            for conversation in split_conversations:
                print('conversation length ::',len(conversation.split(' ')))
                names.append(modelobj.get_names(luke_tokenizer, luke_model,conversation))
            names = [item for sublist in names for item in sublist]

            # print('flat list',names)
            entities_dictionary = {}
            entities_dictionary['SSN']=ssns
            entities_dictionary['MEMBERID']=memberids
            entities_dictionary['EMAIL_ADDRESS']=emailaddresses
            
            entities_dictionary['PHONENUMBER']=phonenumbers
            entities_dictionary['DOB']=dobs
            entities_dictionary['NAME']=names

            print(entities_dictionary)
            entitities.append(ssns)
            entitities.append(memberids)
            entitities.append(emailaddresses)
            entitities.append(phonenumbers)
            entitities.append(dobs)
            entitities.append(names)

            
            for key, values in entities_dictionary.items():
                
                for value in values:
                    input_text = helpersobj.replace_mask_value(input_text,key,value)
                    print(key,value,"--",input_text)
                    print("***********************")

            removed_numbers=cleaningobj.remove_large_numbers(input_text)

            removed_numbers=cleaningobj.remove_numbers(input_text)

        except Exception as exception:

            import traceback
            trace = traceback.extract_tb(exception.__traceback__)
            for frame in trace:
                print("File:", frame.filename, ", Line:", frame.lineno)
            print("Exception in ",inspect.currentframe().f_code.co_name,exception)
            return {"exception":str(exception)}
        
        return removed_numbers