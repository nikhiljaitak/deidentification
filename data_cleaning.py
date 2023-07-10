import re
import inspect
import sys
import re
import time
import logging
from app_enum import AppEnum
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Cleaning:

    def remove_confidentiality_notice(self, text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        pattern = r"CONFIDENTIALITY NOTICE.*?attachments"
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def remove_large_numbers(self, text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        import re
        pattern = "\+?\d{8,}"
        replaced_text=text
        numbers_regex = re.findall(pattern, replaced_text)
        for number in numbers_regex:
            replaced_text=replaced_text.replace(number,'[NUMBER]')
        return replaced_text
    
    def remove_numbers(self, input_text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        pattern = "[0-9/()]"
        output_text = re.sub(pattern, "", input_text)

        return output_text
    
    def clean_text(self, text):
        print(AppEnum.FUNCTION_CURRENT_PLACEHOLDER.value, inspect.currentframe().f_code.co_name)
        print('before total words=',len(text.split()))
        from string import punctuation
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in tokens if word not in stop_words]
        text = ' '.join(words)+' '
        print('total words=',len(text.split()))
        return text