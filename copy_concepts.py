import json
from lxml import etree as ET
import os

def create_qu_output_gold():
    output_file="tmp/ir/output/bioasq_qa_GOLD.xml"
    gold_json = "testing_datasets/BioASQ-training8b/augmented_test_FULL_ABSTRACTS.json"
    with open(gold_json) as f:
        json_obj = json.load(f)
        gold_questions = json_obj.get("questions")
        root = ET.Element("Input")
        n = 1
        for gold_question in gold_questions:
            print(f"question {n}/{len(gold_questions)}")
            id = gold_question.get("id")
            print("id",id)
            question = gold_question.get("body")
            print("question",question)
            qtype = gold_question.get("type")
            q = ET.SubElement(root,"Q")
            q.set('id',str(id))
            q.text = question
            qp = ET.SubElement(q,"QP")
            qp_type = ET.SubElement(qp,'Type')
            qp_type.text = qtype
            ent_list = []
            ent_list = gold_question.get("human_concepts")
            if ent_list:
                print(len(ent_list) ,"concepts found",type(ent_list))
                for ent in ent_list:
                    qp_en = ET.SubElement(qp,'Entities') 
                    qp_en.text = str(ent)
            qp_query = ET.SubElement(qp,'Query')
            if ent_list:
                qp_query.text = str(' '.join(ent_list))
            # Create IR tag
            IR = ET.SubElement(q, "IR")
            print("creating ir ")
        tree = ET.ElementTree(root)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        tree.write(output_file, pretty_print=True)
        print(f"writing XML to {output_file}")

if __name__ == "__main__":
    # copy over the human_concepts into their place in the XML file

    create_qu_output_gold()

