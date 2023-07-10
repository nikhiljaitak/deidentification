from models import Model
import inspect
from app_enum import AppEnum
import re
import phonenumbers
import spacy

class Entities:
    def __init__(self):
        print('init')
    
    def get_member_ids(self, input_text):
        member_ids = Entities.get_member_id(input_text)
        return member_ids

    def get_email_addresses(self, input_text):
        email_addresses = Entities.get_email_address(input_text)
        return email_addresses

    def get_ssns(self, input_text):
        ssns = Entities.get_ssn(input_text)
        return ssns

    def get_phone_numbers(self, input_text):
        phone_numbers = Entities.get_phone_number(input_text)
        return phone_numbers

    
    def get_dobs(self, input_text):
        dobs = Entities.get_dob(input_text)
        return dobs

    def get_names(self, model ,tokenizer, text):
        names = Entities.get_name(self, model ,tokenizer, text)
        return names
    
    @staticmethod
    def get_email_address(input_text):
            print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
            email_regex=AppEnum.REGEX_EMAIL.value
            import re
            match=re.findall(email_regex,input_text)
            if len(match)>0:
                return match
            else:
                return []
    @staticmethod    
    def get_member_id( input_text):
            print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
            memberids=[]
            try:
                import re
                for regex in AppEnum.REGEX_MEMBER_ID.value:
                    regex_match=re.findall(regex, str(input_text).lower())
                    for match in regex_match:
                         memberids.append(re.sub(AppEnum.REGEX_ONLY_DIGITS.value, '', match))
            except Exception as e:
                print('exception+'+str(e),memberids)
            finalmemberids = list(filter(lambda x: len(str(x)) > 6, memberids))
            return finalmemberids
        
    @staticmethod       
    def get_ssn(text):
            print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
            ssn_pattern=AppEnum.REGEX_SSN.value
            import re
            match=re.findall(ssn_pattern,text)
            if len(match)>0:
                return match
            else:
                return []
            
    @staticmethod       
    def get_phone_number(text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        for match in phonenumbers.PhoneNumberMatcher(text, None):
            numbers=[]
            if phonenumbers.is_valid_number(match.number):
                  numbers.append('+'+str(match.number.country_code)+str(match.number.national_number))
            return numbers
        return []
    
    @staticmethod
    def get_dob(text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        dates=[]
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append(ent.text)
        return dates
    
    @staticmethod
    def get_dob_re(input_text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        
        date_formats = [r"\s*\d{1,2}/\d{1,2}/\d{4}\s*", r"\s*\d{4}-\d{1,2}-\d{1,2}\s*", r"\s*\d{1,2}\.\d{1,2}\.\d{4}\s*", r"\s*[A-Z][a-z]{2,8} \d{1,2}, \d{4}\s*", r"\s*\d{1,2} [A-Z][a-z]{2,8} \d{4}\s*",r"\s*\d{4}/\d{1,2}/\d{1,2}\s*","\d{1,2}/\d{1,2}/\d{4}",r"\s*\d{1,2}/\d{1,2}/\d{2}\s*"]
        dates = []
    
        for date_format in date_formats:
            dates += re.findall(date_format, input_text)
    
        return dates
    
    def get_name(self, model ,tokenizer, text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        modelobj=Model()
        names=modelobj.get_names(model ,tokenizer ,text )
        return names
        
       
