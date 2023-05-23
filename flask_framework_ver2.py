from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
import sys
import json
import sys
sys.path.append('/Users/nikhiljaitak/Downloads/CHATGPT/Deidentification')
import time
import deidentification_ver2
from deidentification_ver2 import NamedEntityRecognition
from Summarizer_ver2 import Summarization

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
#lukeModel=modelObj.getLukeModel('/Ensemble_Torch_NER')
#lukeModel=modelObj.loadSavedModel('Ensemble_Torch_NER/Ensemble_Torch_NER/')
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
        text = request.form['input_text']
        task = request.form['task']
        print(text,task)
        response['question']=text

        start_time = time.time()
        deidentified = entityObj.get_entities(lukeTokenizer, lukeModel, text)
        response['deidentified']=deidentified
        end_time = time.time()
        running_time = end_time - start_time
        response['modeltime']=str(running_time)+" seconds"
    
        if task == 'deidentify_summarize_chatgpt':
            start_time = time.time()
            result= entityObj.get_chatgpt_response(deidentified)
            end_time = time.time()
            running_time = end_time - start_time
            response['chatgptApiTime']=str(running_time)+" seconds"
            response['summary_chatgpt']=result
        
        elif task=='deidentify_summarize':
            print('custom model: Summarizer')
            start_time = time.time()
            result = summaryObj.get_summary_bard(deidentified)
            response['summary']=result
            end_time = time.time()
            running_time = end_time - start_time
            response['bardSummarizerTime']=str(running_time)+" seconds"            

    except Exception as e:
        response['error']=e
        print(e)
    
    
    return jsonify(response)
    
if __name__ == '__main__':
    app.run(port=8081)
