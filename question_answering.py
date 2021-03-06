"""
question_answering.py handles passing the response from the qu and ir portions of the pipeline into the BioBERT model so that we can preoperly retrieve an answer to the original question
we utilize run_yesno.py, run_factoid.py, and run_list.py with the proper parameters passed in to accomplish this.

Necessary args = vocab_file, bert_config_file, and output_dir
     
fields for input json for factoid/list/yesno:
 - context is abstract text
{
  "data": [
    "paragraphs": [
      {
      "context": String,
      "qas" : [
        "question" : String,
        "id" : String
        ]
      }
    ]
}]
"""
import warnings
warnings.filterwarnings('ignore')

import json
from json import loads
import os
import shutil
import time
from bs4 import BeautifulSoup as bs

# pass formatted json into file that generates answer
def run_qa_file(filename, output_dir,predict_file):
    print(f"\033[95mRunning {filename}\033[0m")
    vocab_file_path = f'data_modules{os.path.sep}model{os.path.sep}vocab.txt'
    bert_config_file = f'data_modules{os.path.sep}model{os.path.sep}config.json'
    command = f"python {filename} --do_train=False --do_predict=True --vocab_file={vocab_file_path} --bert_config_file={bert_config_file} --output_dir={output_dir} --predict_file={predict_file}"
    print(f"\033[95mRunning command: {command}\033[0m")
    os.system(command)

# prints json to file ;)
def print_json_to_file(file, json_data, batch_mode = False):
    if(batch_mode):
        try:
            with open(file,"r") as old_json_file:
                old_json_data = json.load(old_json_file)
                if (old_json_data):
                    # If there is an old valid json file here, append the data
                    old_data = old_json_data['data']
                    new_data = json_data['data'][0]
                    old_data.append(new_data)
                    json_data['data'] = old_data
        except:
            print("\033[95mFirst run\033[0m")
    with open(file,'w') as outfile:
        json.dump(json_data,outfile,indent=4)
        outfile.close()

# This is all to get the data in the proper format for the json file
def get_json_from_data(data):
    id, type, question, abstract = data
    json_data = {}
    qas = [{'id':id, 'question':question}]
    one_item = {'qas':qas,'context':abstract}
    paragraphs = []
    paragraphs.append(one_item)
    json_data['data'] = [{'paragraphs':paragraphs}]
    return json_data

# Transform the nbest_predictions.json and predictions.json into proper format for the Evaluation Measures repository
def transform_to_bioasq(file_paths):
    factoid_old, list_old, yesno_old = file_paths
    factoid_command = f"python ./biocodes/transform_n2b_factoid.py --nbest_path={factoid_old} --output_path={os.path.dirname(factoid_old)}"
    list_command = f"python ./biocodes/transform_n2b_list.py --nbest_path={list_old} --output_path={os.path.dirname(list_old)}"
    yesno_command = f"python ./biocodes/transform_n2b_yesno.py --nbest_path={yesno_old} --output_path={os.path.dirname(yesno_old)}"
    if os.path.exists(factoid_old):
        print("\033[95mChanging Factoid!\033[0m")
        os.system(factoid_command)
    #commenting this out until List question formatting 
    if os.path.exists(list_old): 
        print("\033[95mChanging List!\033[0m")
        os.system(list_command)
    if os.path.exists(yesno_old):
        print("\033[95mChanging yesno!\033[0m")
        os.system(yesno_command)
    
#ensure temp directory and subdirectories exist
def setup_file_system(output_dir,batch_mode = False):
    tmpdir_path = os.getcwd() + os.path.sep + 'tmp' + os.path.sep
    inputfile_path = output_dir +'qa_input.json'
    out_file_name = 'predictions.json'
    outfile_path = output_dir + out_file_name
    factoid_path = output_dir + "factoid" + os.path.sep
    yesno_path = output_dir + "yesno" + os.path.sep
    list_path = output_dir + "list" + os.path.sep
    if not os.path.isdir(tmpdir_path):
        os.mkdir (tmpdir_path)
    if not os.path.isdir(output_dir):
        os.mkdir (output_dir)
    if batch_mode:
        if not os.path.isdir(factoid_path):
            os.mkdir (factoid_path)
        if not os.path.isdir(yesno_path):
            os.mkdir (yesno_path)
        if not os.path.isdir(list_path):
            os.mkdir (list_path)
    return inputfile_path, outfile_path, factoid_path,yesno_path,list_path

def get_answer(json_data, output_dir, batch_mode = False):
    id, type, question,abstract = json_data
    inputfile_path,outfile_path,factoid_path,yesno_path,list_path = setup_file_system(output_dir)
    # list nbest is used to respond with multiple results
    if(batch_mode):
        factoid_file_path = factoid_path + "qa_factoids.json"
        yesno_file_path = yesno_path + "qa_yesno.json"
        list_file_path = list_path + "qa_list.json"
        printing_json = get_json_from_data(json_data)
        if type == 'yesno':
            print_json_to_file(yesno_file_path, printing_json, batch_mode=True)
        elif type == 'factoid':
            print_json_to_file(factoid_file_path, printing_json, batch_mode=True)
        elif type == 'list':
            print_json_to_file(list_file_path, printing_json, batch_mode=True)
        else: # We don't handle the summary case
            return 
    else:
        print(f'\033[95mQuestion answering json: {json_data}\033[0m ')
        # Write data in BioASQ format to json file
        good_json_data = get_json_from_data(json_data)
        print_json_to_file(inputfile_path, good_json_data)
        print(f"\033[95mQuestion type <{type}>\033[0m")
        if type == 'yesno':
            run_qa_file('run_yesno.py',output_dir, predict_file=inputfile_path)
        elif type == 'factoid':
            run_qa_file('run_factoid.py',output_dir, predict_file=inputfile_path)
        elif type == 'list':
            run_qa_file('run_list.py',output_dir, predict_file=inputfile_path)
            list_nbest = output_dir + "nbest_predictions.json"
            # allow for getting multiple predictions
            outfile_path = list_nbest
        else: # We don't handle the summary case
            return 
        while (not os.path.exists(outfile_path)):
            time.sleep(1)
        # Wait for qa script to finish to respond with answer if not batch mode
        if os.path.isfile(outfile_path):
            with open(outfile_path,'r') as j:
                results = json.loads(j.read()) 
                j.close()
                return results

def run_batch_mode(input_file,output_dir):
    print(f"\033[95mreading {input_file} for input\033[0m")
    with open(input_file, "rU") as file:
        content = file.readlines()
        content = "".join(content)
        soup = bs(content,"lxml")
        result = soup.find_all("q") # get all the questions
        for item in result:
            type = item.find("qp").find("type").get_text()
            id = item.attrs['id']
            original_question = str(item.find('qp').previousSibling)
            try:
                abstract_text = item.find('ir').find('result').find("abstract").get_text()
            except:
                # If IR was unsuccessful when it came to retrieving documents for the given question
                abstract_text = ""
            data = (id, type, original_question, abstract_text)
            print(f"\033[95mGetting answer for \'{original_question}\'\033[0m")
            # write all questions to a general file
            json_data = get_json_from_data(data)
            print_json_to_file(output_dir+ "qa_all.json", json_data, batch_mode=True)
            if abstract_text != "":
                # get the answers for questions with relevant concepts
                get_answer(data,output_dir,batch_mode=True)
    
    # Now that the intermediary files are generated, pass them into qa scripts. 
    _,_,factoid_path,yesno_path,list_path = setup_file_system(output_dir,True)

    factoid_file_path = factoid_path + "qa_factoids.json"
    yesno_file_path = yesno_path + "qa_yesno.json"
    list_file_path = list_path + "qa_list.json"
    
    list_nbest = list_path+"nbest_predictions.json"
    factoid_nbest = factoid_path+"nbest_predictions.json"
    # We use predictions instead of nbest since yesno only has 2 options
    yesno_preds = yesno_path+"predictions.json" 
    # Run the biobert question answering code on our extracted question dataframes
    run_qa_file('run_yesno.py',yesno_path, predict_file= yesno_file_path)
    run_qa_file('run_factoid.py',factoid_path, predict_file= factoid_file_path)
    run_qa_file('run_list.py',list_path, predict_file= list_file_path)
    
    # Run the nbest predictions through a file type transformer, then into BioASQ evaluation repo
    while not os.path.exists(list_nbest):
        time.sleep(1)
    if os.path.isfile(list_nbest):
        print("\033[95mMigrating jsons to correct bioasq format!!\033[0m")
        file_paths = (factoid_nbest, list_nbest, yesno_preds)
        transform_to_bioasq(file_paths)




