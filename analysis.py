from re import S
from lxml import etree as ET

import numpy as np
import pandas as pd
import json
import os
import csv
import subprocess

"""
    for QU we are doing f1 score on concepts
    for IR we are doing f1 score on document ids
    for QA we use the BioASQ testing repo
"""

# make master json for eval purposes
def generate_master_golden_json(filenames):
    input_file = open('/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B1_golden.json')
    master_json = json.load(input_file)
    for json_location in filenames:
        json_file = open(json_location)
        raw_json = json.load(json_file)
        for question in raw_json['questions']:
            master_json['questions'].append(question)
        json_file.close()
    input_file.close()
    output_file = open("/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/master_golden.json","w")
    json.dump(master_json,output_file,indent=2)

# This is so that our system can actually use the golden test datapoints..... of course we can't test with training questions, there would be no overlap.
def create_testing_csv():
    input_file = open('/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/master_golden.json')
    master_json = json.load(input_file)
    input_file.close()

    output_csv = open('/home/daniels/dev/BioASQ-QA-System/testing_datasets/evaluation_input.csv',"w")
    csv_writer = csv.writer(output_csv)
    csv_writer.writerow(['ID','Question'])
    for question in master_json['questions']:
        csv_writer.writerow([question['id'],question['body']])
    output_csv.close()


def run_evaluation_code(file_to_evaluate):
    # now to check our stats from the evaluation measures repo (user will need to write in path to repo)
    # java -Xmx10G -cp flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 "/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B1_golden.json" "/home/daniels/dev/BioASQ-QA-System/tmp/qa/yesno/BioASQform_BioASQ-answer.json" -verbose
    eval_measures_repo_path = "/home/daniels/dev/Evaluation-Measures"
    golden_file_path = "/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/master_golden.json"
    path_to_jar = "/home/daniels/dev/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    evaluation_process = subprocess.Popen(
        [
            "java",
            "-Xmx10G",
            "-cp",
            path_to_jar,
            "evaluation.EvaluatorTask1b",
            "-phaseB",
            "-e",
            "5",
            golden_file_path,
            file_to_evaluate,
            "-verbose",
        ],
        cwd=eval_measures_repo_path,
        stdout=subprocess.PIPE,
    )
    stdout, _ = evaluation_process.communicate()
    return stdout.decode("utf-8")


def f1_score(predicted, actual):
    # precision = true predicted positives / all predicted positives
    # recall = true predicted positives/ all actual positives
    # f1 = 2 * (precision * recall) / (precision + recall)
    true_positives = [
        ele for ele in predicted if ele in actual
    ]  # correctly predicted positives
    false_positives = np.setdiff1d(
        predicted, actual
    )  # elements that were predicted incorrectly
    false_negatives = np.setdiff1d(
        actual, predicted
    )  # important elements which were not predicted
    if len(predicted) == 0:
        precision = 0
        recall = 0
    else:
        precision = len(true_positives) / len(predicted)
        recall = len(true_positives) + len(false_negatives)
    if precision == 0 or recall == 0: # shorctut
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def get_generated_dict(file_location, input):
    dict = {}
    with open(file_location, "r") as xml_file:
        fileTree = ET.parse(xml_file)
        if fileTree:
            root = fileTree.getroot()
            questions = root.findall("Q")
            for question in questions:
                qid = question.get("id")
                if input:
                    qp = question.find("QP")
                    entities = [e.text for e in qp.findall("Entities")]
                    dict[qid] = entities
                else:
                    ir = question.find("IR")
                    result_pmids = [e.get("PMID") for e in ir.findall("Result")]
                    dict[qid] = result_pmids
    return dict


def get_scores(gold_dict, gen_dict):
    f1_scores = {}
    for key in gold_dict.keys():
        f1_scores[key] = f1_score(gen_dict[key], gold_dict[key])
    return f1_scores


def get_gold_dicts():
    # Here we are going to be opening files and retrieving a dict of features with keys taken from question IDs
    golden_dataset = "testing_datasets/BioASQ-training8b/training8b.json"
    qu_dict = {}
    ir_dict = {}
    with open(golden_dataset, "r") as file:
        empty = 0
        found = 0
        data = json.load(file)
        questions = data["questions"]
        for question in questions:
            id = question.get("id")
            human_concepts = question.get("human_concepts")
            if not human_concepts:
                human_concepts = []
                empty += 1
            else:
                found += 1
            documents = [
                document.split("/")[-1] for document in question.get("documents")
            ]
            qu_dict[id] = human_concepts
            ir_dict[id] = documents
    print(f"\033[95mConcepts -> empty: {empty} | found: {found}\033[0m")
    return qu_dict, ir_dict


if __name__ == "__main__":
    '''
        Various dataset manipulation scripts will be left in comments here
    '''
    # files=['/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B2_golden.json','/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B3_golden.json','/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B4_golden.json','/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B5_golden.json']
    # generate_master_golden_json(files)

    # print("\033[31m -- Analysis Module --\033[0m")
    # qu_gold, ir_gold = get_gold_dicts()
    # qu_generated = get_generated_dict("tmp/ir/input/bioasq_qa.xml", True)
    # ir_generated = get_generated_dict("tmp/ir/output/bioasq_qa.xml", False)
    # qu_f1 = get_scores(gold_dict=qu_gold, gen_dict=qu_generated)
    # ir_f1 = get_scores(gold_dict=ir_gold, gen_dict=ir_generated)
    # qu_f1_sum = 0.0
    # ir_f1_sum = 0.0

    # for key in qu_f1.keys():
    #     qu_f1_sum += qu_f1.get(key)
    # for key in ir_f1.keys():
    #     ir_f1_sum += ir_f1.get(key)
    # qu_f1_sum /= len(qu_f1)
    # ir_f1_sum /= len(ir_f1)

    # print(f"\033[95mAverage QU f1 score:\n{qu_f1_sum}\033[0m")
    # print(f"\033[95mAverage IR f1 score:\n{ir_f1_sum}\033[0m")

    # # QA module analysis
    # yesno_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa/yesno/BioASQform_BioASQ-answer.json"
    # factoid_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa/factoid/BioASQform_BioASQ-answer.json"
    # list_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa/list/BioASQform_BioASQ-answer.json"
    # print("\n\n\033[95mQA module evaluation\033[0m")
    # print(run_evaluation_code(yesno_file))
    # print(run_evaluation_code(factoid_file))
    # print(run_evaluation_code(list_file))
    # print("\033[31mEvaluation complete.\033[0m")
    create_testing_csv()



