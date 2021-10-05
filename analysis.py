from re import L, S
import re
from lxml import etree as ET

import numpy as np
import pandas as pd
import json
import os
import csv
import subprocess
from sklearn import metrics as m

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


def eval_score(predicted, actual):
    # precision = true predicted positives / all predicted positives
    # recall = true predicted positives/ all actual positives
    # f1 = 2 * (precision * recall) / (precision + recall)
    # force lowercase to help out here
    predicted = set([ele.lower() for ele in predicted])
    actual = set([ele.lower() for ele in actual])

    # correctly predicted positives
    true_positives = [
        ele for ele in predicted if ele in actual
    ]  
    # elements that were predicted incorrectly
    false_positives = [
        ele for ele in predicted if ele not in actual
    ]
    # important elements which were not predicted
    false_negatives = [
        ele for ele in actual if ele not in predicted
    ]

    if len(predicted) == 0 or len(actual) == 0:
        precision = 0
        recall = 0
    else:
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        recall = len(true_positives) / (len(true_positives)+ len(false_negatives))
    if precision == 0 or recall == 0: # shortcut
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    print(f"\n---\nF1: [{len(predicted)} predicted] {predicted} | [{len(actual)} actual] {actual} \n{len(true_positives)} correct answers : {true_positives}\n{len(false_positives)} incorrect answers: {false_positives}\n{len(false_negatives)} missed answers: {false_negatives}\n")
    print(f"f1: {f1}, precision: {precision}, recall: {recall}")
    return (f1,precision,recall)


def get_generated_dict(file_location, mode="concepts"):
    dict = {}
    with open(file_location, "r") as xml_file:
        fileTree = ET.parse(xml_file)
        if fileTree:
            root = fileTree.getroot()
            questions = root.findall("Q")
            print(f"{len(questions)} {mode} found")
            for question in questions:
                qid = question.get("id")
                if mode=="concepts":
                    qp = question.find("QP")
                    concepts = [e.text for e in qp.findall("Entities")] # entities are the same as concepts
                    dict[qid] = concepts
                elif mode=="pubmed_ids":
                    ir = question.find("IR")
                    result_pmids = [e.get("PMID") for e in ir.findall("Result")]
                    dict[qid] = result_pmids
                elif mode =="type":
                    qp = question.find("QP")
                    result_type = qp.find("Type").text
                    dict[qid] = (result_type,question.text)
    return dict

def get_generated_dicts(generated_xml):
    print(f"generated xml file is {generated_xml}")
    print("Getting generated concepts")
    generated_concepts = get_generated_dict(generated_xml, mode="concepts")
    print("Getting generated pmids")
    generated_pubmed_ids = get_generated_dict(generated_xml, mode="pubmed_ids")
    print("Getting generated types")
    generated_types = get_generated_dict(generated_xml, mode="type")
    return generated_concepts,generated_pubmed_ids,generated_types

# f1_scores takes the form {id: (f1,precision,recall)}
def get_scores(gold_dict, gen_dict):
    eval_scores = {}
    for key in gold_dict.keys():
        # only take duplicates to not penalize for hitting same document multiple times
        eval_scores[key] = eval_score(set(gen_dict[key]), set(gold_dict[key]))
    return eval_scores

def get_gold_dicts(golden_dataset):
    # Here we are going to be opening files and retrieving a dict of features with keys taken from question IDs
    concepts_dict = {}
    pubmed_ids_dict = {}
    type_dict = {}
    with open(golden_dataset, "r") as file:
        empty = 0
        found = 0
        data = json.load(file)
        questions = data["questions"]
        for question in questions:
            id = question.get("id")
            question_type = question.get("type")
            human_concepts = question.get("human_concepts")
            if not human_concepts:
                empty += 1
            else:
                found += 1
            documents = [
                document.split("/")[-1] for document in question.get("documents")
            ]
            # print(f"documents found: for question ({id}) : {len(documents)}")
            if human_concepts:
                concepts_dict[id] = human_concepts
            else:
                concepts_dict[id] = []
            pubmed_ids_dict[id] = documents
            type_dict[id] = (question_type,question["body"])
    print(f"\033[95mConcepts -> empty: {empty} | found: {found}\033[0m")
    print(f" {len(concepts_dict)} concepts {len(pubmed_ids_dict)} ids {len(type_dict)} types")
    return concepts_dict, pubmed_ids_dict, type_dict

def get_average_scores(f1_list):
    if len(f1_list) == 0:
        return
    f1_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    for key in f1_list.keys():
        f1_sum += f1_list.get(key)[0]
        precision_sum += f1_list.get(key)[1]
        recall_sum += f1_list.get(key)[2]

    return (f1_sum / len(f1_list),  precision_sum / len(f1_list), recall_sum / len(f1_list))

def exact_matching(gold, generated):
    num_correct = 0
    for id in generated:
        if id in gold:
            if generated[id][0] == gold[id][0]:
                num_correct+=1
            # else:
                # print(f"Question [{id}], {generated[id][1]}, Guessed <{generated[id][0]}> when it should have been <{gold[id][0]}>")
        else:
            print(f"question {id} not in evalation set")
    return f"{num_correct}/{len(generated)}"

def print_qu_ir_results(EVALUATING = False, TESTING = False):

    print("\033[31m -- Analysis Module --\033[0m")

    if EVALUATING or TESTING:
        golden_dataset = "testing_datasets/Task8BGoldenEnriched/master_golden.json"
    else:
        golden_dataset = "testing_datasets/BioASQ-training8b/training8b.json"
    if TESTING:
        generated_xml = "tmp/debugging/generated_ir.xml"
    elif EVALUATING:
        generated_xml = "tmp/ir/output/bioasq_qa_EVAL.xml"
    else:
        generated_xml= "tmp/ir/output/bioasq_qa.xml"
    print("Getting golden data")
    gold_concepts, gold_pubmed_ids, gold_question_types = get_gold_dicts(golden_dataset)
    print(f"\033[31mNum gold concepts: {len(gold_concepts)}, pmids: {len(gold_pubmed_ids)}, types: {len(gold_question_types)}\033[0m")

    generated_concepts, generated_pubmed_ids, generated_types= get_generated_dicts(generated_xml)
    print(f"\033[31mNum generated concepts: {len(generated_concepts)}, pmids: {len(generated_pubmed_ids)}, types: {len(generated_types)}\033[0m")
    print("\033[95mgetting f1 scores\033[0m")
    concepts_scores = get_scores(gold_dict=gold_concepts, gen_dict=generated_concepts)

    pubmed_scores = get_scores(gold_dict=gold_pubmed_ids, gen_dict=generated_pubmed_ids)
    type_em = exact_matching(gold=gold_question_types,generated=generated_types)

    qu_concepts_scores_average = get_average_scores(concepts_scores)
    ir_scores_average = get_average_scores(pubmed_scores)

    print(f"\033[95mAverage QU concepts f1, precision, recall score:\n{qu_concepts_scores_average}\033[0m")
    print(f"\033[95mNumber of question's with type correctly predicted {type_em}\033[0m")
    print(f"\033[95mAverage IR PMID f1, precision, recall score:\n{ir_scores_average}\033[0m")

def print_qa_results(TESTING=False,EVALUATING=False):
    # QA module analysis
    if TESTING:
        yesno_file= "/home/daniels/dev/BioASQ-QA-System/tmp/debugging/generated_yesno.json"
        factoid_file= "/home/daniels/dev/BioASQ-QA-System/tmp/debugging/generated_factoid.json"
        list_file= "/home/daniels/dev/BioASQ-QA-System/tmp/debugging/generated_list.json"
    elif EVALUATING:
        yesno_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa_EVAL/yesno/BioASQform_BioASQ-answer.json"
        factoid_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa_EVAL/factoid/BioASQform_BioASQ-answer.json"
        list_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa_EVAL/list/BioASQform_BioASQ-answer.json"
    else:
        yesno_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa/yesno/BioASQform_BioASQ-answer.json"
        factoid_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa/factoid/BioASQform_BioASQ-answer.json"
        list_file= "/home/daniels/dev/BioASQ-QA-System/tmp/qa/list/BioASQform_BioASQ-answer.json"

    print("\n\n\033[95mQA module evaluation\033[0m")
    print(run_evaluation_code(yesno_file))
    print(run_evaluation_code(factoid_file))
    print(run_evaluation_code(list_file))
    print("\033[31mEvaluation complete.\033[0m")

if __name__ == "__main__":
    '''
        Various dataset manipulation scripts will be left in comments here
    '''
    # files=['/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B2_golden.json','/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B3_golden.json','/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B4_golden.json','/home/daniels/dev/BioASQ-QA-System/testing_datasets/Task8BGoldenEnriched/8B5_golden.json']
    # generate_master_golden_json(files)
    # create_testing_csv()
    EVALUATING = False
    TESTING = False

    print_qu_ir_results(TESTING=TESTING, EVALUATING=EVALUATING)
    #print_qa_results(TESTING=TESTING, EVALUATING=EVALUATING)    



