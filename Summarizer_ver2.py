from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration
from transformers import pipeline

class Summarization:

    def get_summary_bard(self, text):
        import re
        text1 = re.sub('[^a-zA-Z0-9]', ' ', text)
        text1 = text1.lower()
        text1 = text1.split()
        text = ' '.join(text1)
        summarizer = pipeline("summarization")
        t_len=len(text.split(' ') )
        summ=summarizer(text, max_length=t_len, min_length=int(t_len/2), do_sample=False)
        return summ[0]['summary_text']
   
    def getSummarization(self, text):
        IS_CNNDM = True 
        LOWER = False
# Load our model checkpoints
        if IS_CNNDM:
            model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
            tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
        else:
            model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased')
            tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

        max_length = 1024 if IS_CNNDM else 512
# generation example
        if LOWER:
            article = text.lower()
        else:
            article = text
            inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
# Generate Summary
            summary_ids = model.generate(inputs["input_ids"])
            summary=tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(summary)
            return summary

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
        


ARTICLE_TO_SUMMARIZE = '''A hunter had a beautiful hound that always went hunting with him. The faithful dog was a fast friend of the hunter’s only son who loved to play with the dog. The master also loved the hound for its friendship with his young son who was only ten years old. The master of the house had no one else in the house.
One day, the hunter went out hunting but forgot to take the hound with him. His son was still asleep in his bed. As he reached the edge of the forest he found that he had forgotten to bring the hound with him. He decided to go back and bring the hound. He came home but was afraid to see his hound all bloodstained standing at the gate. The hunter thought that the hound had killed his son. He whipped out his sword and killed the dog. Inspirational Moral Stories for Adults
He quickly walked into the house and saw blood pools here and there. Just then, he saw his son coming out of his room. The boy told his Father that a wolf had come into the house and was about to kill him when the hound pounced upon it and tore it to pieces. The hunter began to cry at his haste in killing the faithful creature.'''
        
#objSum = Summarization()
#objSum.get_chatgpt_response(ARTICLE_TO_SUMMARIZE)
#objSum.get_summary_bard(ARTICLE_TO_SUMMARIZE)


 