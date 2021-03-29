
import os
import subprocess
import json
from lxml import etree as ET
from collections import OrderedDict
import re
"""
grep "GO:0005488" MRCONSO.RRF
C1167622|ENG|S|L0216497|VC|S0415665|N|A11582442|||GO:0005488|GO|PT|GO:0005488|binding|0|N|256|
C1749457|ENG|S|L0023688|PF|S0289396|Y|A14251462|||GO:0005488|GO|ET|GO:0005488|ligand|0|N||


PF > VC
Preferred Variant

PT > ET > SY
Preferred Entry Synonym

"""

def get_plaintext_from_umls(uid):
    get_concepts_regex = "(?=[|]*)+([A-z -])+(?=[|]\d[|]\w*[|]\d*[|])"
    grep_out_path = "tmp/grep_out.txt"

    command_str = f"grep '{uid}' umls/MRCONSO.RFF"
    print('command:',command_str)
    output = subprocess.check_output(command_str,shell=True)
    #os.system(command_str)
    print(output)

    with open(grep_out_path,"r") as file:
        concepts = re.findall(get_concepts_regex,file.read())
    if concepts:
        return concepts


# pull the relavant concepts from their original format ex: ("http://amigo.geneontology.org/cgi-bin/amigo/term_details?term=GO:0005154") -> ("GO:0005154")
def get_readable_concepts(dirty_concept):
    get_uid_regex = "(GO:\d{7})|(D\d{6})"
    result = re.search(get_uid_regex,dirty_concept)
    clean_uid = result.group()
    print(dirty_concept, clean_uid)
    return get_plaintext_from_umls(clean_uid)

def remove_duplicates(concepts):
    return list(OrderedDict.fromkeys(concepts))

def make_human_readable(concepts):
    human_readable_concepts = []
    for concept in concepts:
        concepts_from_db = get_readable_concepts(concept)
        human_readable_concepts.extend(concepts_from_db)
    return remove_duplicates(human_readable_concepts)

# This script is designed to extract human-readable terms from a UMLS representation such as GO:0005488 -> [ligand, binding]
if __name__ == "__main__":

    #path_to_json = f"testing_datasets{os.path.sep}BioASQ-training8b{os.path.sep}training8b.json"
    path_to_json2 = f"tmp{os.path.sep}sample.json"

    # 1) get a reference to the training8b.json
    training_dataset = None
    with open(path_to_json2) as file:
        training_dataset = json.loads(file.read())

    # 2) loop through each element and get the human readable text for each concept
    questions = training_dataset["questions"]
    for question in questions:
        print("****")
        concepts = question["concepts"]
        human_concepts = make_human_readable(concepts)
        question["human_concepts"] = human_concepts
    
    # 3) place human readable concepts in human_concepts in the same json object


