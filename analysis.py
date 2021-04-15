from lxml import etree as ET

import numpy as np
import pandas as pd
import json
import os

"""
    for QU we are doing f1 score on concepts
    for IR we are doing f1 score on document ids
    for QA we use the BioASQ testing repo
"""

def f1_score(predicted, actual):
    # precision = true predicted positives / all predicted positives
    # recall = true predicted positives/ all actual positives
    # f1 = 2 * (precision * recall) / (precision + recall)
    true_positives=[ele for ele in predicted if ele in actual] # correctly predicted positives
    false_positives=np.setdiff1d(predicted,actual)  # elements that were predicted incorrectly
    missed_positives=np.setdiff1d(actual,predicted) # important elements which were not predicted
    if len(predicted) == 0:
        precision = 0
        recall = 0
    else:
        precision = len(true_positives) / len(predicted)
        recall = len(true_positives) + len (missed_positives)
    if precision == 0 or recall == 0:
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
            questions = root.findall('Q')
            for question in questions:
                qid = question.get("id")
                if input:
                    qp = question.find('QP')
                    entities = [e.text for e in qp.findall("Entities")]
                    dict[qid] = entities
                else:
                    ir = question.find('IR')
                    result_pmids = [e.get("PMID") for e in ir.findall("Result")]
                    dict[qid] = result_pmids
    return dict

def get_scores(gold_dict, gen_dict):
    f1_scores = {}
    for key in gold_dict.keys():
        f1_scores[key] = f1_score(gen_dict[key],gold_dict[key])
    return f1_scores

def get_gold_dicts():
    #Here we are going to be opening files and retrieving a dict of features with keys taken from question IDs
    golden_dataset = "testing_datasets/BioASQ-training8b/training8b.json"
    qu_dict = {}
    ir_dict = {}
    with open(golden_dataset, "r") as file:
        empty = 0
        found = 0
        data = json.load(file)
        questions = data["questions"]
        for question in questions:
            id = question.get('id')
            human_concepts = question.get('human_concepts')
            if not human_concepts:
                human_concepts = []
                empty +=1
            else:
                found+=1
            documents = [document.split("/")[-1] for document in question.get('documents')]
            qu_dict[id] = human_concepts
            ir_dict[id] = documents
    print(f"Concepts -> empty: {empty} | found: {found}")
    return qu_dict,ir_dict

if __name__ == "__main__":
    print(" -- Analysis Module --")
    qu_gold,ir_gold = get_gold_dicts()
    qu_generated = get_generated_dict("tmp/ir/input/bioasq_qa.xml",True)
    ir_generated = get_generated_dict("tmp/ir/output/bioasq_qa.xml",False)
    qu_f1 = get_scores(gold_dict=qu_gold,gen_dict=qu_generated)
    ir_f1 = get_scores(gold_dict=ir_gold,gen_dict=ir_generated)
    qu_f1_sum = 0.0
    ir_f1_sum = 0.0

    for key in qu_f1.keys():
        qu_f1_sum += qu_f1.get(key)
    for key in ir_f1.keys():
        ir_f1_sum += ir_f1.get(key)
    qu_f1_sum/=len(qu_f1)
    ir_f1_sum/=len(ir_f1)

    print(f"Average QU f1 score:\n{qu_f1_sum}")
    print(f"Average IR f1 score:\n{ir_f1_sum}")


