
import os
import subprocess
import json
from lxml import etree as ET
from collections import OrderedDict
import re
from googletrans import Translator
"""
grep "GO:0005488" MRCONSO.RRF
C1167622|ENG|S|L0216497|VC|S0415665|N|A11582442|||GO:0005488|GO|PT|GO:0005488|binding|0|N|256|
C1749457|ENG|S|L0023688|PF|S0289396|Y|A14251462|||GO:0005488|GO|ET|GO:0005488|ligand|0|N||

Example split line = ['C1167622', 'ENG', 'S', 'L0216497', 'VC', 'S0415665', 'N', 'A11582442', '', '', 'GO:0005488', 'GO', 'PT', 'GO:0005488', 'binding', '0', 'N', '256', '']

PF > VC
Preferred Variant

PT > ET > SY
Preferred Entry Synonym

PT = Designated Preferred name

We should only pull PT representations of each mesh id and use it as the unique identifier for the MESH ID since there should only be ONE Preferred Variant.

"""

def force_english(unknown_text):
    trans = Translator()
    translated_text = trans.translate(text=unknown_text, to_lang='en').text
    return translated_text

def get_plaintext_from_umls(uid):
    get_concepts_regex = "(?=[|]*)+([\w\d\s\- \, \.])+(?=[|]\d[|]\w*[|]\d*[|])"
    grep_out_path = "tmp/grep_out.txt"
    command_str = f"grep 'PT|{uid}' umls/MRCONSO.RRF" 
    print(command_str)
    try:
        output = subprocess.check_output(command_str,shell=True).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        return None
    concepts_iter = re.finditer(get_concepts_regex,output)

    good_concept = None
    for concept in concepts_iter:
        good_concept = concept.group()
    good_concept = force_english(good_concept)
    print(f"\033[33m{good_concept}\033[95m")
    return good_concept

# pull the relavant concepts from their original format ex: ("http://amigo.geneontology.org/cgi-bin/amigo/term_details?term=GO:0005154") -> ("GO:0005154")
def get_readable_concepts(dirty_concept):
    get_uid_regex = "(GO:\d{7})|(D\d{6})"
    result = re.search(get_uid_regex,dirty_concept)
    if(result):
        clean_uid = result.group()
        return get_plaintext_from_umls(clean_uid)
    else:
        return None

def remove_duplicates(concepts):
    return list(OrderedDict.fromkeys(concepts))

def make_human_readable(concepts):
    human_readable_concepts = []
    for concept in concepts:
        concepts_from_db = get_readable_concepts(concept)
        if(concepts_from_db):
            human_readable_concepts.append(concepts_from_db)
    return remove_duplicates(human_readable_concepts)

# This script is designed to extract human-readable terms from a UMLS representation such as GO:0005488 -> [ligand, binding]
if __name__ == "__main__":
    path_to_json = f"testing_datasets{os.path.sep}BioASQ-training8b{os.path.sep}training8b.json"
    path_to_json2 = f"tmp{os.path.sep}sample.json"

    # 1) get a reference to the training8b.json
    training_dataset = None
    with open(path_to_json) as file:
        training_dataset = json.loads(file.read())

    # 2) loop through each element and get the human readable text for each concept
    questions = training_dataset["questions"]
    n = 1
    for question in questions:
        print(f"\033[95mQuestion {n}: {question.get('body')}\033[0m")
        concepts = question.get("concepts")
        # if(concepts and question.get("human_concepts") != None):
        if concepts:
            print("concepts:",concepts,"\n\n")
            human_concepts = make_human_readable(concepts)
            question["human_concepts"] = human_concepts
        n=n+1
    # 3) write changes back to file

    with open(path_to_json,'w') as outfile:
        json.dump(training_dataset,outfile,indent=4)
        outfile.close()
