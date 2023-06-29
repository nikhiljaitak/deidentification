from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask import Flask
import sys
import json
#sys.path.append('/Users/nikhiljaitak/Downloads/CHATGPT/Deidentification')
#sys.path.append('/home/ubuntu/deisummarizer_git/deidentification')
import time
import deidentification
from deidentification import NamedEntityRecognition

app = Flask(__name__)

entityObj = NamedEntityRecognition()

#lukeTokenizer=entityObj.getLukeTokenizer()
#lukeModel=entityObj.getLukeModel()
#entityObj.saveHuggingFaceModel( 'Models/Ensemble_Torch_NER/Tokenizer/',lukeTokenizer)
#entityObj.saveHuggingFaceModel( 'Models/Ensemble_Torch_NER/',lukeModel)

#lukeTokenizer=entityObj.loadTokenizer('/Users/nikhiljaitak/Downloads/Suresh_Sir_/podman_deidentifier/app/Models/Ensemble_Torch_NER/Tokenizer/')

lukeTokenizer=entityObj.loadTokenizer('Ensemble_Torch_NER/Tokenizer/')

lukeModel=entityObj.loadSavedModel('Ensemble_Torch_NER/')

#lukeModel=entityObj.loadSavedModel('/Users/nikhiljaitak/Downloads/Suresh_Sir_/podman_deidentifier/app/Models/Ensemble_Torch_NER/')

from flask_cors import cross_origin

@app.route('/')
def home():
    return 'Application health status" running'


@app.route('/messages', methods=['POST'])
def model():
    response={}
    try:

        print("request is ",request.form)
        input_text = request.form['input_text']
        task = request.form['task']
        uniqueid = request.form['uniqueid']
        cgpt_endpoint = request.form['cgpt_endpoint']
        max_Tokens = request.form['max_Token']
        num_Responses = request.form['num_Responses']
        temperature = request.form['temperature']
        regen_Temperature = request.form['regen_Temperature']
        api_key = request.form['api_key']
        history_conversations = request.form['history_conversations']

        # cgpt_endpoint = 'https://api.openai.com/v1/engines/text-davinci-003/completions'
        # max_Tokens = 500
        # num_Responses = 2
        # temperature = 0.7
        # regen_Temperature = 0.7
        # api_key = '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'
        # history_conversations = ''

        print(input_text,task,cgpt_endpoint,max_Tokens, num_Responses, temperature,regen_Temperature)
        
        '''

        from kafka import KafkaProducer
        from json import dumps
        print('yes')
        producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))

        data = {'input_text' : input_text,'id':uniqueid}
        print("date ",data)
        producer.send('test-topic', value=data)
        '''

        response['question']=input_text
        response['cgpt_endpoint']=cgpt_endpoint
        response['maxToken']=max_Tokens
        response['numResponses']=num_Responses
        response['temperature']=temperature
        response['regenTemperature']=regen_Temperature
        response['apikey']=api_key
        response['history_conversations']=history_conversations
        response['uniqueid']=uniqueid

        chatgpt_configs={'cgptEndpoint':cgpt_endpoint,'maxTokens':str(max_Tokens),
                         'numResponses':num_Responses,'temperature':temperature,
                         'regenTemperature':regen_Temperature,'apiKey':api_key,
                         'history_conversations':history_conversations}

        start_time = time.time()
        deidentified = entityObj.get_entities(lukeTokenizer, lukeModel, input_text)
        response['deidentified']=deidentified
        end_time = time.time()
        running_time = end_time - start_time
        response['modeltime']=str(running_time)+" seconds"
    
        if task == 'deidentify_chatgpt':
            start_time = time.time()
            result= entityObj.get_chatgpt_response(deidentified, chatgpt_configs)
            end_time = time.time()
            running_time = end_time - start_time
            response['chatgptApiTime']=str(running_time)+" seconds"
            response['chatgpt']=result
        
        elif task=='deidentify_summarize':
            print('custom model: Summarizer')
            start_time = time.time()
            result = summaryObj.get_summary_bard(deidentified)
            response['summary']=result
            end_time = time.time()
            running_time = end_time - start_time
            response['bardSummarizerTime']=str(running_time)+" seconds"    

        elif task=='deidentify_chatgpt_classification':
            print('Deidentify and Classification of Email')
            start_time = time.time()
            classificationPrompt="Assign one of the following categories to the text: Rewards Issue, Password Reset, Inquiry, Locked Issue  and others "
            result = entityObj.get_chatgpt_response(classificationPrompt+"::"+deidentified, chatgpt_configs)
            response['classification']=result
            end_time = time.time()
            running_time = end_time - start_time
            response['answerTime']=str(running_time)+" seconds"     

    except Exception as e:
        response['error']=e
        print(e)
    
    print(response)
    return jsonify(response)
    
if __name__ == '__main__':
    app.run()
