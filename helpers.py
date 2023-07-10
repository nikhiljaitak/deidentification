import nltk
import math
class Helpers:

    def split_conversations(self, text):
        words = nltk.word_tokenize(text)
        conversations = []
        start_token=0
        max_tokens=400
        temp_max=max_tokens
        print("total conversation:",math.ceil(len(words)/max_tokens),"   total length ::",len(words))
        for iteration in range(math.ceil(len(words)/max_tokens)):
            conversations.append(' '.join(words[start_token:max_tokens]))
            start_token=max_tokens
            max_tokens=max_tokens+temp_max
        return conversations
    
    def replace_mask_value(self, text, replacevalue, to_mask ):
        masked_text = text.replace(to_mask,'['+replacevalue+']')
        return masked_text