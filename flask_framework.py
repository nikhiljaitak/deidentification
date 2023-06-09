from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
import sys
import json
import sys
sys.path.append('/Users/nikhiljaitak/Downloads/CHATGPT/Deidentification')
#sys.path.append('/home/ubuntu/deisummarizer_git/deidentification')
import time
import deidentification
from deidentification import NamedEntityRecognition
from Summarizer import Summarization

app = Flask(__name__)

# Set the log level (optional)
app.logger.setLevel(logging.ERROR)

# Create a file handler that rotates log files
handler = RotatingFileHandler('error.log', maxBytes=10240, backupCount=10)

# Set the log format
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
handler.setFormatter(formatter)

# Add the file handler to the Flask logger
app.logger.addHandler(handler)



CORS(app)

@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error('An error occurred: %s', error)
    return ('An error occurred.',error)

entityObj = NamedEntityRecognition()
summaryObj=Summarization()
lukeTokenizer=entityObj.loadTokenizer('/Users/nikhiljaitak/Downloads/Suresh_Sir_/Codes/Dev/Code_27Sept/SO_AI-ML_L1L2EmailIntentPredictor-master/Models/Ensemble_Torch_NER/Tokenizer/')

#lukeTokenizer=entityObj.loadTokenizer('/home/ubuntu/deisummarizer/Weights/Ensemble_Torch_NER/Tokenizer/')
#lukeModel=entityObj.loadSavedModel('/home/ubuntu/deisummarizer/Weights/Ensemble_Torch_NER/')

lukeModel=entityObj.loadSavedModel('/Users/nikhiljaitak/Downloads/Suresh_Sir_/Codes/Dev/Code_27Sept/SO_AI-ML_L1L2EmailIntentPredictor-master/Models/Ensemble_Torch_NER/')

from flask_cors import cross_origin
@app.route('/')
def home():
    return 'proper address please'

@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
@app.route('/messages', methods=['POST'])
def model():
    response={}
    try:
        print(request)
        input_text = request.form['input_text']
        task = request.form['task']
        cgpt_endpoint = request.form['cgpt_endpoint']
        max_Tokens = request.form['max_Token']
        num_Responses = request.form['num_Responses']
        temperature = request.form['temperature']
        regen_Temperature = request.form['regen_Temperature']
        api_key = request.form['api_key']
        history_conversations = request.form['history_conversations']
        print(input_text,task,cgpt_endpoint,max_Tokens, num_Responses, temperature,regen_Temperature)

        
        response['question']=input_text
        response['cgpt_endpoint']=cgpt_endpoint
        response['maxToken']=max_Tokens
        response['numResponses']=num_Responses
        response['temperature']=temperature
        response['regenTemperature']=regen_Temperature
        response['apikey']=api_key
        response['history_conversations']=history_conversations

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
    
    
    return jsonify(response)
    
if __name__ == '__main__':
    app.run(port=8081)
