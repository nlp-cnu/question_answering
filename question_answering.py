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
import time
from bs4 import BeautifulSoup as bs

# pass formatted json into file that generates answer
def run_qa_file(filename, output_dir,predict_file):
    print(f"Running {filename}")
    vocab_file_path = f'data_modules{os.path.sep}model{os.path.sep}vocab.txt'
    bert_config_file = f'data_modules{os.path.sep}model{os.path.sep}config.json'
    command = f"python {filename} --do_train=False --do_predict=True --vocab_file={vocab_file_path} --bert_config_file={bert_config_file} --output_dir={output_dir} --predict_file={predict_file}"
    os.system(command)

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
            print("First run")
    with open(file,'w') as outfile:
        json.dump(json_data,outfile,indent=4)
        outfile.close()

def get_json_from_data(data):
    id, type, question, abstract = data
    # This is all to get the data in the proper format for the json file
    json_data = {}
    qas = [{'id':id, 'question':question}]
    one_item = {'qas':qas,'context':abstract}
    paragraphs = []
    paragraphs.append(one_item)
    json_data['data'] = [{'paragraphs':paragraphs}]
    return json_data

def setup_file_system(output_dir):
    tmpdir_path = os.getcwd() + os.path.sep + 'tmp' + os.path.sep
    inputfile_path = 'tmp'+os.path.sep+'qa_input.json'
    out_file_name = 'predictions.json'
    outfile_path = output_dir + out_file_name
    factoid_path = output_dir + "factoid" + os.path.sep
    yesno_path = output_dir + "yesno" + os.path.sep
    list_path = output_dir + "list" + os.path.sep
    if not os.path.isdir(tmpdir_path):
        os.mkdir (tmpdir_path)
    if not os.path.isdir(output_dir):
        os.mkdir (output_dir)
    if not os.path.isdir(factoid_path):
        os.mkdir (factoid_path)
    if not os.path.isdir(yesno_path):
        os.mkdir (yesno_path)
    if not os.path.isdir(list_path):
        os.mkdir (list_path)
    return inputfile_path, outfile_path, factoid_path,yesno_path,list_path

def get_answer(json_data, output_dir, batch_mode = False):
    inputfile_path,outfile_path,factoid_path,yesno_path,list_path = setup_file_system(output_dir)
    if(batch_mode):
        factoid_file_path = factoid_path + "qa_factoids.json"
        yesno_file_path = yesno_path + "qa_yesno.json"
        list_file_path = list_path + "qa_list.json"
    else:
        print ('qa_data', json_data)
    if(batch_mode):
        if type == 'yesno':
            print_json_to_file(yesno_file_path, json_data, batch_mode=True)
        elif type == 'factoid':
            print_json_to_file(factoid_file_path, json_data, batch_mode=True)
        elif type == 'list':
            print_json_to_file(list_file_path, json_data, batch_mode=True)
        else: # We don't handle the summary case
            return 
    else:
        # Write data to json file
        print_json_to_file(inputfile_path, json_data)
        print("finished writing results")
        if type == 'yesno':
            run_qa_file('run_yesno.py',output_dir, predict_file=inputfile_path)
        elif type == 'factoid':
            run_qa_file('run_factoid.py',output_dir, predict_file=inputfile_path)
        elif type == 'list':
            run_qa_file('run_list.py',output_dir, predict_file=inputfile_path)
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
    print(f"reading {input_file} for input")
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
            print(f"Getting answer for \'{original_question}\'")
            # write all questions to a general file
            json_data = get_json_from_data(data)
            print_json_to_file(output_dir+ "qa_all.json", json_data, batch_mode=True)
            if abstract_text != "":
                # get the answers for questions with relevant concepts
                get_answer(json_data,output_dir,batch_mode=True)
    
    # Now that the intermediary files are generated, pass them into qa scripts. 
    factoid_path = output_dir + "factoid" + os.path.sep
    yesno_path = output_dir + "yesno" + os.path.sep
    list_path = output_dir + "list" + os.path.sep
    factoid_file_path = factoid_path + "qa_factoids.json"
    yesno_file_path = yesno_path + "qa_yesno.json"
    list_file_path = list_path + "qa_list.json"
    print("Running predictions")

    run_qa_file('run_yesno.py',yesno_path, predict_file= yesno_file_path)
    run_qa_file('run_factoid.py',factoid_path, predict_file= factoid_file_path)
    run_qa_file('run_list.py',list_path, predict_file= list_file_path)

def clear_tmp_dir(files):
    print("tmp dir cleaning")
    for file in files:
        os.remove(file)
