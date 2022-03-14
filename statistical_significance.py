## this is for statistical significance only
from numpy import isin
from lxml import etree as et
import pandas as pd
import json
from utils import *

DEBUG = False

"""
THESE ARE COPIED OVER FROM analysis.py so that I can tweak some values for excempting list from the stats significance tests.
"""


def get_three_files(a_dir):
    return [
        a_dir + "/factoid/predictions.json",
        # a_dir + "/list/predictions.json",
        a_dir
        + "/yesno/predictions.json",
    ]


def get_answers(answers_files):
    all_answers = dict()
    for a in answers_files:
        with open(a, "r") as f:
            d = json.loads(f.read())
            if DEBUG:
                print(f"{len(d)} answers found in {a}")
            for key, value in d.items():
                if key in all_answers.keys():
                    print(f"MULTIPLE ANSWERS FOR {key}")
                if isinstance(value, list):
                    all_answers[key] = value[
                        0
                    ]  # get the first value which is answer not prediction for the yes/no
                else:
                    all_answers[key] = value
    return all_answers


def parse_xml(xml_file, dir_for_qa):
    no_answers = 0
    # get answers
    qa_answers = get_answers(get_three_files(dir_for_qa))
    # get ir and qu
    df_cols = [
        "id",
        "human_concepts",
        "documents",
        "full_abstracts",
        "titles",
        "type",
        "exact_answer",
    ]
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    for question in xroot:
        id = question.attrib.get("id")
        ir = question.find("IR")
        qp = question.find("QP")
        concepts = [e.text for e in qp.findall("Entities")]
        qa_type = qp.find("Type").text
        titles = [e.find("Title").text for e in ir.findall("Result")]
        abstracts = [e.find("Abstract").text for e in ir.findall("Result")]
        pmids = [e.get("PMID") for e in ir.findall("Result")]
        exact_answer = qa_answers[id] if id in qa_answers else None
        if DEBUG and not exact_answer:
            print(f"id [{id}] has no answer")
            no_answers += 1
        rows.append(
            {
                "id": id,
                "human_concepts": concepts,
                "documents": pmids,
                "full_abstracts": abstracts,
                "titles": titles,
                "type": qa_type,
                "exact_answer": exact_answer,
            }
        )
    out_df = pd.DataFrame(rows, columns=df_cols)
    if DEBUG:
        print(
            f"{GREEN}[{no_answers}/{len(out_df)}]{OFF} {WHITE}questions had answers{OFF}"
        )
    return out_df


def get_pmid(docs):
    documents = [document.split("/")[-1] for document in docs]
    return documents


def get_col_list(gold_df, gen_df, col):
    gold_col = gold_df.loc[:, ["id", col]].copy()
    gen_col = gen_df.loc[:, ["id", col]].copy()

    gold = gold_col.to_dict(orient="list")
    gen = gen_col.to_dict(orient="list")
    gen_ids = gen["id"]
    gen_vals = gen[col]
    gold_ids = gold["id"]
    gold_vals = gold[col]
    return gold_ids, gold_vals, gen_ids, gen_vals


def get_gold_df(gold_dataset_path):
    with open(gold_dataset_path, "r") as f:
        gold_data = json.loads(f.read())
    # load and flatten data
    gold_df = pd.json_normalize(gold_data, record_path="questions")
    # get gold df
    gold_df["documents"] = gold_df["documents"].apply(get_pmid)
    return gold_df


def do_sig_formatting(gold_df, gen_df, file_name, q_type=None):
    if q_type:
        typed_gold_df = gold_df[gold_df["type"] == q_type]
        typed_gen_df = gen_df[gen_df["type"] == q_type]
        gold_df = typed_gold_df
        gen_df = typed_gen_df

    gold_ids, gold_ans, gen_ids, gen_ans = get_col_list(gold_df, gen_df, "exact_answer")
    save_sig_file(gold_ids, gold_ans, gen_ids, gen_ans, file_name, q_type)


def save_sig_file(gold_ids, gold_ans, gen_ids, gen_ans, name_of_file, q_type=None):
    # print(f"{name_of_file} | {YELLOW}gold_ids:{len(gold_ids)}, gold_ans:{len(gold_ans)}, gen_ids:{len(gen_ids)}, gen_ans:{len(gen_ans)}{OFF}")
    # list of 1 for match or 0 for not
    count_correct = 0
    matches = []

    # handle factoid ids
    for i, gen_id in enumerate(gen_ids):
        if len(gen_id) == 24:
            gen_ids[i] = gen_id[0:20]
    for i, gold_id in enumerate(gold_ids):
        if len(gold_id) == 24:
            gold_ids[i] = gold_id[0:20]
    

    for i in range(len(gold_ids)):
        gold_val = gold_ans[i]
        if isinstance(gold_val,list):
            gold_val = gold_val[0]
        try:
            gen_val = gen_ans[gen_ids.index(gold_ids[i])]
        except ValueError as e:
            if DEBUG:
                print(e)
            matches.append(0)
            continue
        if gen_val:
            if gold_val == gen_val:
                matches.append(1)
                count_correct += 1
            else:
                matches.append(0)
        else:
            matches.append(0)
    if not q_type:
        q_type = "factoid and yesno"
    print(f"{count_correct}/{len(matches)} {q_type} questions answered correctly.")
    with open(name_of_file, "w") as file:
        for m in matches:
            file.write(f"0 1 {m}\n")


if __name__ == "__main__":
    """
    Get every dataset and create a file for significance testing
    The file should have a row for each question in the dataset
    format should be
    0 1 1 if gen matches gold
    else
    0 1 0 if gen doesnt match gold
    """

    golden_dataset_path = "testing_datasets/augmented_concepts_abstracts_titles.json"
    gen_folder = "tmp/submit_to_dr_henry/datasets"
    gold_df = get_gold_df(golden_dataset_path)

    systems = ["gen_abs", "gen_snips", "gold_abs", "gold_snips"]
    names = [
        "bioasq_qa_GENERATED_ABSTRACTS.xml",
        "bioasq_qa_GENERATED_SNIPPETS.xml",
        "bioasq_qa_GOLD_ABSTRACTS.xml",
        "bioasq_qa_GOLD_SNIPPETS.xml",
    ]

    for i, s in enumerate(systems):
        system_folder = f"{gen_folder}/{s}"
        ir_dataset = f"{system_folder}/{names[i]}"
        # set file names
        name_for_sig_file = ir_dataset.replace(".xml", "_SIGNIFICANCE_TESTING.txt")
        name_for_sig_file_factoid = name_for_sig_file.replace(".txt", "_factoid.txt")
        name_for_sig_file_yesno = name_for_sig_file.replace(".txt", "_yesno.txt")
        print(f"{CYAN}Opening dataset: {ir_dataset}{OFF}")
        gen_df = parse_xml(ir_dataset, f"{system_folder}")
        # do all then seperate into factoid and yesno
        # ALL
        do_sig_formatting(gold_df, gen_df, name_for_sig_file)
        # factoid
        do_sig_formatting(gold_df, gen_df, name_for_sig_file_factoid, q_type="factoid")
        # yesno
        do_sig_formatting(gold_df, gen_df, name_for_sig_file_yesno, q_type="yesno")
