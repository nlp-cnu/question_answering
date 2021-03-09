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
import json
import os
import time

def run_qa_file(filename, output_dir,predict_file):
    print(f"Running {filename}")
    vocab_file_path = f'data_modules{os.path.sep}model{os.path.sep}vocab.txt'
    bert_config_file = f'data_modules{os.path.sep}model{os.path.sep}config.json'
    command = f"python {filename} --do_train=False --do_predict=True --vocab_file={vocab_file_path} --bert_config_file={bert_config_file} --output_dir={output_dir} --predict_file={predict_file}"
    os.system(command)

def get_answer(data, output_dir):
    print ('qa_data', data)
    inputfile_path = '/tmp/qa_input.json'
    tmpdir_path = os.getcwd() + os.path.sep + 'tmp' + os.path.sep
    out_file_name = 'predictions.json'
    outfile_path = output_dir + out_file_name

    id, type, question, abstract = data
    # This is all to get the data in the proper format for the json file
    qas = [{'id':id, 'question':question}]
    one_item = {'qas':qas,'context':abstract}
    paragraphs = []
    paragraphs.append(one_item)
    json_data = {}
    json_data['data'] = [{'paragraphs':paragraphs}]

    if not os.path.isdir(tmpdir_path):
        os.mkdir (tmpdir_path)
    #Write data to json file
    with open(inputfile_path,'w') as outfile:
        json.dump(json_data,outfile,indent=4)
        outfile.close()
    if type == 'yesno':
        run_qa_file('run_yesno.py',output_dir, predict_file=outfile_path)
    elif type == 'factoid':
        run_qa_file('run_factoid.py',output_dir, predict_file=outfile_path)
    elif type == 'list':
        run_qa_file('run_list.py',output_dir, predict_file=outfile_path)
    else:
        print("This should never log....")
        return 
    # Wait for qa script to finish to respond with answer

    while (not os.path.exists(outfile_path)):
        time.sleep(1)
    if os.path.isfile(outfile_path):
        with open(outfile_path,'r') as j:
            results = json.loads(j.read()) 
            j.close()
            return results

def clear_tmp_dir():
    print("tmp dir cleaning")
