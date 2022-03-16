from asyncio.subprocess import DEVNULL
from lxml import etree as et
from sklearn.metrics import classification_report
import numpy
import os
import pandas as pd
import json
import time

from utils import *

DEBUG = False


def get_pmid(docs):
    documents = [document.split("/")[-1] for document in docs]
    return documents


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


def get_three_files(a_dir):
    return [
        a_dir + "/factoid/predictions.json",
        a_dir + "/list/predictions.json",
        a_dir + "/yesno/predictions.json",
    ]


def get_col_list(gold_df, gen_df, col):
    gold_col = gold_df.loc[:, ["id", col]].copy()
    if gen_df == None:
        gen_ids = None
        gen_vals = None
    else:
        gen_col = gen_df.loc[:, ["id", col]].copy()


    gold = gold_col.to_dict(orient="list")
    gen = gen_col.to_dict(orient="list")
    gen_ids = gen["id"]
    gen_vals = gen[col]
    gold_ids = gold["id"]
    gold_vals = gold[col]
    return gold_ids, gold_vals, gen_ids, gen_vals


def parse_qu_output(xml_file):
    df_cols = [
        "id",
        "human_concepts",
        "type",
    ]
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    for question in xroot:
        id = question.attrib.get("id")
        qp = question.find("QP")
        concepts = [e.text for e in qp.findall("Entities")]
        qa_type = qp.find("Type").text
        rows.append(
            {
                "id": id,
                "human_concepts": concepts,
                "type": qa_type,
            }
        )
    return pd.DataFrame(rows, columns=df_cols)


def parse_ir_output(xml_file):
    df_cols = [
        "id",
        "documents",
    ]
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    for question in xroot:
        id = question.attrib.get("id")
        ir = question.find("IR")
        pmids = [e.get("PMID") for e in ir.findall("Result")]
        rows.append(
            {
                "id": id,
                "documents": pmids,
            }
        )
    return pd.DataFrame(rows, columns=df_cols)


def parse_qa_output(xml_file, qa_folder):
    # get answers
    qa_answers = get_answers(get_three_files(qa_folder))
    df_cols = [
        "id",
        "type",
        "exact_answer",
    ]
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    for question in xroot:
        id = question.attrib.get("id")
        qp = question.find("QP")
        qa_type = qp.find("Type").text
        exact_answer = qa_answers[id] if id in qa_answers else None
        rows.append(
            {
                "id": id,
                "type":qa_type,
                "exact_answer": exact_answer,
            }
        )
    return pd.DataFrame(rows, columns=df_cols)


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


def print_yes_no_info(df, tag):
    print(tag)
    print(f" [{len(df)}] {tag} Yes/No Questions")
    yes_df = df[df["exact_answer"] == "yes"]
    no_df = df[df["exact_answer"] == "no"]
    print(f" [{len(yes_df)}] {tag} Yes Questions")
    print(f" [{len(no_df)}] {tag} No Questions")


""" f1 Yes
    tp is gen 'yes' | gold 'yes'
    fp is gen 'yes' | gold 'no'
    fn is gen 'no' |  gold 'yes'

    f1 No
    tp is gen 'no' | gold 'no'
    fp is gen 'no' | gold 'yes'
    fn is gen 'yes' |  gold 'no'

    IGNORE if the predicted type is yes/no but gold type is different
"""


def do_yes_no_eval(gold_df, gen_df):
    print(f"{CYAN}Yes/No Evaluation{OFF}")
    yes_no_gold_df = gold_df[gold_df["type"] == "yesno"]
    yes_no_gen_df = gen_df[gen_df["type"] == "yesno"]

    if DEBUG:
        # Gold stats
        print_yes_no_info(yes_no_gold_df, "Gold")
        # Gen Stats
        print_yes_no_info(yes_no_gen_df, "Generated")

    gold_ids, gold_ans, gen_ids, gen_ans = get_col_list(
        yes_no_gold_df, gen_df, "exact_answer"
    )

    # YES
    ytp = 0
    yfp = 0
    yfn = 0

    for i in range(len(gold_ids)):
        gold_val = gold_ans[i]
        try:
            gen_val = gen_ans[gen_ids.index(gold_ids[i])]
        except ValueError as e:
            if DEBUG:
                print(e)
            continue
        if gen_val:
            if gold_val == "yes":
                if gen_val == "yes":
                    ytp += 1
                elif gen_val == "no":
                    yfn += 1
                else:
                    if DEBUG:
                        print(
                            f"yes question [{gold_ids[i]}] had generated answer {gen_val}"
                        )
            elif gold_val == "no":
                if gen_val == "yes":
                    yfp += 1
                elif gen_val == "no":
                    pass  # handled by no f1
                else:
                    if DEBUG:
                        print(
                            f"no question [{gold_ids[i]}] had generated answer {gen_val}"
                        )
            else:
                print(
                    f"GOLDEN answer to yes/no question [{gold_ids[i]}] was {gold_val}"
                )

        else:  # not identified as yes/no question by generated
            pass
    # sanity check
    print(
        f"Yes | True Positive: {GREEN}{ytp}{OFF}, False Positive: {GREEN}{yfp}{OFF}, False Negative: {GREEN}{yfn}{OFF}"
    )
    try:
        yp = ytp / (ytp + yfp)
    except:
        yp = 0
    try:
        yr = ytp / (ytp + yfn)
    except:
        yr = 0
    try:
        yf1 = 2 * ((yp * yr) / (yp + yr))
    except:
        yf1 = 0
    print(
        f"Yes | f1 {GREEN}{yf1}{OFF}, precision {GREEN}{yp}{OFF}, recall {GREEN}{yr}{OFF}"
    )

    # NO SIDE
    ntp = 0
    nfp = 0
    nfn = 0

    for i in range(len(gold_ids)):
        gold_val = gold_ans[i]
        try:
            gen_val = gen_ans[gen_ids.index(gold_ids[i])]
        except ValueError as e:
            if DEBUG:
                print(e)
            continue
        if gen_val:
            if gold_val == "no":
                if gen_val == "no":
                    ntp += 1
                elif gen_val == "yes":
                    nfn += 1
                else:
                    if DEBUG:
                        print(
                            f"no question [{gold_ids[i]}] had generated answer {gen_val}"
                        )
            elif gold_val == "yes":
                if gen_val == "no":
                    nfp += 1
                elif gen_val == "yes":
                    pass  # handled by no f1
                else:
                    print(
                        f"yes question [{gold_ids[i]}] had generated answer {gen_val}"
                    )
            else:
                if DEBUG:
                    print(
                        f"GOLDEN answer to yes/no question [{gold_ids[i]}] was {gold_val}"
                    )

        else:  # not identified as yes/no question by generated
            pass

    # sanity check
    print(
        f"No | True Posative: {GREEN}{ntp}{OFF}, False Posative: {GREEN}{nfp}{OFF}, False Negative: {GREEN}{nfn}{OFF}"
    )
    try:
        np = ntp / (ntp + nfp)
    except:
        np = 0
    try:
        nr = ntp / (ntp + nfn)
    except:
        nr = 0
    try:
        nf1 = 2 * ((np * nr) / (np + nr))
    except:
        nf1 = 0
    print(
        f"No | f1 {GREEN}{nf1}{OFF}, precision {GREEN}{np}{OFF}, recall {GREEN}{nr}{OFF}"
    )

    f1 = (yf1 + nf1) / 2
    p = (yp + np) / 2
    r = (yr + nr) / 2
    print(
        f"Overall Yes/No | f1 {GREEN}{f1}{OFF}, precision {GREEN}{p}{OFF}, recall {GREEN}{r}{OFF}"
    )
    print("\n")
    return yf1, yp, yr, nf1, np, nr, f1, p, r


# yes_no_report = do_yes_no_eval(gold_df,gen_df)
# Compute [[Mean average precision, Geometric mean average precision]], precision, recall, f1 score
def do_concepts_eval(gold_df, gen_df):
    print(f"{CYAN}Concepts Evaluation{OFF}")
    gold_ids, gold_cons, gen_ids, gen_cons = get_col_list(
        gold_df, gen_df, "human_concepts"
    )
    num_gen_q_without_cons = 0
    num_gold_q_without_cons = 0
    tp = 0
    fp = 0
    fn = 0

    scores = []
    # for each question
    for i in range(len(gold_ids)):
        gold_val = gold_cons[i]
        if not isinstance(gold_val, list) or gold_val == []:
            num_gold_q_without_cons += 1
            continue
        try:
            gen_val = gen_cons[gen_ids.index(gold_ids[i])]
        except ValueError as e:
            if DEBUG:
                print(e)
            continue
        # if concepts are found
        if gen_val != []:
            # TP is concept in Gold AND Gen
            # FP is concept NOT IN GOLD, but YES IN GEN
            # FN is concept IN Gold but NOT GEN

            # get unique concepts from both gold and gen
            unique_gold_cons = set(gold_val[0])
            unique_gen_cons = set(gen_val[0])
            for val in unique_gold_cons:
                if val in unique_gen_cons:
                    tp += 1
                elif val not in unique_gen_cons:
                    fn += 1
            for val in unique_gen_cons:
                if val not in unique_gold_cons:
                    fp += 1

            f1, p, r = get_f1_p_r(tp, fp, fn, tag="Concepts")
            scores.append((f1, p, r))
        else:  # There are no concepts retrieved for this document
            num_gen_q_without_cons += 1
            pass
    # sanity check
    print(
        f"{GREEN}[{len(gold_ids) - num_gold_q_without_cons}/{len(gold_ids)}]{OFF} Questions have human readable concepts in gold dataset"
    )
    print(
        f"{GREEN}[{len(gen_ids) - num_gen_q_without_cons}/{len(gen_ids)}]{OFF} Questions have human readable concepts in generated dataset"
    )

    # OVERALL SCORES
    f1_sum = p_sum = r_sum = 0
    for f1, p, r in scores:
        f1_sum += f1
        p_sum += p
        r_sum += r
    f1_sum /= len(scores)
    p_sum /= len(scores)
    r_sum /= len(scores)

    print(
        f"Concepts mean f1 {GREEN}{f1_sum}{OFF}, precision {GREEN}{p_sum}{OFF}, recall {GREEN}{r_sum}{OFF}"
    )
    print("\n")
    return f1_sum, p_sum, r_sum, scores


# concepts_report = do_concepts_eval(gold_df,gen_df)
def get_f1_p_r(tp, fp, fn, tag="calculated"):
    if DEBUG:
        print(f"{tag} tp: {tp}, fp: {fp}, fn: {fn}")
    try:
        p = tp / (tp + fp)
    except:
        p = 0
    try:
        r = tp / (tp + fn)
    except:
        r = 0
    try:
        f1 = 2 * ((p * r) / (p + r))
    except:
        f1 = 0
    if DEBUG:
        print(f"{tag} f1 {f1}, precision {p}, recall {r}")
    return f1, p, r


def do_pmids_eval(gold_df, gen_df):
    print(f"{CYAN}PubMed Documents Evaluation{OFF}")
    # pmids are the pubmed document ids
    gold_ids, gold_pmids, gen_ids, gen_pmids = get_col_list(
        gold_df, gen_df, "documents"
    )
    num_gen_q_without_docs = 0
    num_gold_q_without_docs = 0
    tp = 0
    fp = 0
    fn = 0

    scores = []
    # for each question
    for i in range(len(gold_ids)):
        gold_val = gold_pmids[i]
        if gold_val == []:
            num_gold_q_without_docs += 1
            continue
        try:
            gen_val = gen_pmids[gen_ids.index(gold_ids[i])]
        except ValueError as e:
            if DEBUG:
                print(e)
            continue
        # if documents are found
        if isinstance(gen_val, list) and gen_val != []:
            # TP is pmid in Gold AND Gen
            # FP is pmid NOT IN GOLD, but YES IN GEN
            # FN is pmid IN Gold but NOT GEN

            # get unique PMIDs from both gold and gen
            unique_gold_pmids = set(gold_val[0])
            unique_gen_pmids = set(gen_val[0])
            for val in unique_gold_pmids:
                if val in unique_gen_pmids:
                    tp += 1
                elif val not in unique_gen_pmids:
                    fn += 1
            for val in unique_gen_pmids:
                if val not in unique_gold_pmids:
                    fp += 1

            f1, p, r = get_f1_p_r(tp, fp, fn, tag="PubMed Documents")
            scores.append((f1, p, r))
        else:  # There are no documents retrieved for this document
            num_gen_q_without_docs += 1
            pass
    # sanity check
    print(
        f"{GREEN}[{len(gold_ids) - num_gold_q_without_docs}/{len(gold_ids)}]{OFF} Questions have documents in gold dataset"
    )
    print(
        f"{GREEN}[{len(gen_ids) - num_gen_q_without_docs}/{len(gen_ids)}]{OFF} Questions have documents in generated dataset"
    )

    # OVERALL SCORES
    f1_sum = p_sum = r_sum = 0
    for f1, p, r in scores:
        f1_sum += f1
        p_sum += p
        r_sum += r
    f1_sum /= len(scores)
    p_sum /= len(scores)
    r_sum /= len(scores)

    print(
        f"PubMed Documents mean f1 {GREEN}{f1_sum}{OFF}, precision {GREEN}{p_sum}{OFF}, recall {GREEN}{r_sum}{OFF}"
    )
    print("\n")
    return f1_sum, p_sum, r_sum, scores


# pmid_report = do_pmids_eval(gold_df,gen_df)
# We use strict and lenient accuracy  (first result, or any result)
def do_factoid_eval(gold_df, gen_df, gen_factoid_path):
    print(f"{CYAN}Factoid Evaluation{OFF}")
    factoid_gold_df = gold_df[gold_df["type"] == "factoid"]
    # factoid_gen_df = gen_df[gen_df["type"] == "factoid"]

    # if DEBUG:
    #     print(f" [{len(factoid_gold_df)}] Gold Factoid Questions")
    #     print(f" [{len(factoid_gen_df)}] Generated Factoid Questions")
    gold_ids, gold_ans, gen_ids, gen_ans = get_col_list(
        factoid_gold_df, gen_df, "exact_answer"
    )

    # Use alternative strategy to handle ranked factoid preds
    with open(gen_factoid_path, "r") as ft_file:
        factoid_gen_json = json.load(ft_file)

    gen_factoid_answers = {}
    for question in factoid_gen_json["questions"]:
        id = question["id"]
        if len(id) == 24:
            id = id[0:20]
        answer = question["exact_answer"]
        if answer == []:
            answer = "empty"
        if isinstance(answer, list):
            if isinstance(answer[0], list):  # handle list in list
                answer = [e[0] for e in answer]
        gen_factoid_answers[id] = answer

    num_gold_q_without_ans = 0
    num_strict = 0
    num_lenient = 0
    num_total = 0
    mrrs = []
    # for each question
    for i in range(len(gold_ids)):
        gold_val = gold_ans[i][0]
        if gold_val == []:
            num_gold_q_without_ans += 1
            continue
        # trim last 4 digits which get removed for the final bioasq form answers
        trimmed_id = gold_ids[i][0:20]
        if trimmed_id not in gen_factoid_answers.keys():
            if DEBUG:
                print(f"{trimmed_id} wasn't correctly identified as factoid")
            continue
        gen_vals = gen_factoid_answers[trimmed_id]
        # do some answer cleaning (helps with whitespace)
        gen_vals_clean = [e.lower().strip() for e in gen_vals]
        if DEBUG:
            print(gold_val, " | ", gen_vals)
        # accuracy calculations
        gold_val_clean = gold_val
        num_total += 1
        if (
            gold_val_clean == gen_vals_clean[0]
        ):  # force lowercase / strip whitespace to help
            num_strict += 1
            num_lenient += 1
        elif gold_val_clean in gen_vals_clean:
            num_lenient += 1

        # mrr calculations
        mrr = 0
        r = 0
        n = len(gen_vals_clean)
        for i in range(1, n + 1):
            if gen_vals_clean[i - 1] == gold_val_clean:
                r = i
                break
        if r != 0:
            mrr = 1 / n * 1 / r
            if DEBUG:
                print(f"{trimmed_id} MRR: {mrr}")
        mrrs.append(mrr)

    average_mrr = sum(mrrs) / len(mrrs)
    lenient_acc = num_lenient / num_total
    strict_acc = num_strict / num_total

    # sanity check
    print(
        f"{GREEN}[{len(gold_ids) - num_gold_q_without_ans}/{len(gold_ids)}]{OFF} Factoid questions have answers in gold dataset"
    )
    print(
        f"{GREEN}[{num_total}/{len(gen_factoid_answers)}]{OFF} Factoid questions have answers in generated dataset"
    )
    print(
        f"Mean Reciprocal Rank (MRR): {GREEN}{average_mrr}{OFF}, Strict Accuracy: {GREEN}{strict_acc}{OFF}, Lenient Accuracy: {GREEN}{lenient_acc}{OFF},"
    )
    print("\n")
    return lenient_acc, strict_acc, average_mrr, mrrs


def do_list_eval(gold_df, gen_df,gen_list_path):
    print(f"{CYAN}List Evaluation{OFF}")
    list_gold_df = gold_df[gold_df["type"] == "list"]
    list_gen_df = gen_df[gen_df["type"] == "list"]

    if DEBUG:
        print(f" [{len(list_gold_df)}] Gold List Questions")
        print(f" [{len(list_gen_df)}] Generated List Questions")

    gold_ids, gold_ans, gen_ids, gen_ans = get_col_list(
        list_gold_df, gen_df, "exact_answer"
    )

    # handle the top 5 values for each list as list of answers
    with open(gen_list_path, "r") as ft_file:
        list_gen_json = json.load(ft_file)

    gen_list_answers = {}
    for question in list_gen_json["questions"]:
        qid = question["id"]
        if len(qid) == 24:
            qid = qid[0:20]
        answer = question["exact_answer"]
        if answer == []:
            answer = "empty"
        if isinstance(answer, list) and isinstance(answer[0], list):  # handle list in list
            answer = [e[0] for e in answer]
        gen_list_answers[qid] = answer

    num_gold_q_without_ans = 0
    tp = 0
    fp = 0
    fn = 0

    num_gen = 0
    scores = []
    # for each question
    for i in range(len(gold_ids)):
        gold_val = gold_ans[i]
        if gold_val == []:
            num_gold_q_without_ans += 1
            continue
        try:
            trimmed_id = gold_ids[i][0:20]
        except ValueError as e:
            if DEBUG:
                print(e)
            continue

        if trimmed_id not in gen_list_answers.keys():
            if DEBUG:
                print(f"{trimmed_id} wasn't correctly identified as factoid")
            continue
        gen_vals = gen_list_answers[trimmed_id]
        gen_vals_clean = [e.lower().strip() for e in gen_vals]
        # if answers are found
        if gen_vals != None: 
            # TP is answer in Gold AND Gen
            # FP is answer NOT IN GOLD, but YES IN GEN
            # FN is answer IN Gold but NOT GEN
            gold_list = gold_val[0]
            for val in gold_list:
                if val in gen_vals_clean:
                    tp += 1
                elif val not in gen_vals_clean:
                    fn += 1
            for val in gen_vals_clean:
                if val not in gold_list:
                    fp += 1
            num_gen += 1
            f1, p, r = get_f1_p_r(tp, fp, fn, tag="List Questions")
            scores.append((f1, p, r))
    # sanity check
    print(
        f"{GREEN}[{len(gold_ids) - num_gold_q_without_ans}/{len(gold_ids)}]{OFF} List questions have answers in gold dataset"
    )
    print(
        f"{GREEN}[{num_gen}/{len(list_gen_df)}]{OFF} List questions have answers in generated dataset"
    )

    # OVERALL SCORES
    f1_sum = p_sum = r_sum = 0
    for f1, p, r in scores:
        f1_sum += f1
        p_sum += p
        r_sum += r
    f1_sum /= len(scores)
    p_sum /= len(scores)
    r_sum /= len(scores)

    print(
        f"List Questions mean f1 {GREEN}{f1_sum}{OFF}, precision {GREEN}{p_sum}{OFF}, recall {GREEN}{r_sum}{OFF}"
    )
    print("\n")
    return f1_sum, p_sum, r_sum, scores


def gen_gold_qu_output(gold_df, gen_folder, xml_name="bioasq_qa.xml"):
    qu_generated = gen_folder + "/ir/input/" + xml_name
    new_file_name = qu_generated.replace(".xml", "_GOLD.xml")

    fileTree = et.parse(qu_generated)
    if fileTree:
        root = fileTree.getroot()
        questions = root.findall("Q")
        for question in questions:
            id = question.attrib.get("id")
            qp = question.find("QP")
            # remove type and entities
            qp.clear()
            # type is the fifth element in the row
            gold_type = gold_df.loc[gold_df["id"] == id].values[0][4]
            type_ele = et.SubElement(qp, "Type")
            type_ele.text = gold_type
            gold_concepts = gold_df.loc[gold_df["id"] == id].values[0][7]
            ent_list = []
            qp_query = et.SubElement(qp, "Query")
            if not isinstance(gold_concepts, list):
                if DEBUG:
                    print(
                        f"Question [{id}] has no golden human_concepts [{type(gold_concepts)}]"
                    )
                continue
            for ent in gold_concepts:
                ent_list.append(str(ent))
                qp_en = et.SubElement(qp, "Entities")
                qp_en.text = str(ent)
            qp_query.text = str(" ".join(ent_list))
        tree = et.ElementTree(root)
        os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
        print(f"Writing gold QU output / IR input to {new_file_name}")
        tree.write(new_file_name, pretty_print=True)
    return new_file_name


def gen_gold_ir_output(gold_df, gen_folder, xml_name="bioasq_qa.xml"):
    ir_generated = gen_folder + "/ir/output/" + xml_name
    new_file_name = ir_generated.replace(".xml", "_GOLD_ABSTRACTS.xml")

    file_tree = et.parse(ir_generated)
    if file_tree:
        root = file_tree.getroot()
        questions = root.findall("Q")
        for question in questions:
            q_id = question.attrib.get("id")
            original_question = question.text
            if DEBUG:
                print(original_question)
            ir = question.find("IR")
            # remove original generated articles
            ir.clear()

            gold_abstracts = gold_df.loc[gold_df["id"] == q_id].values[0][8]
            gold_titles = gold_df.loc[gold_df["id"] == q_id].values[0][9]
            # system just using top abstract atm
            if isinstance(gold_abstracts, list) and gold_abstracts != []:
                gold_abstract = gold_abstracts[0]
            else:
                gold_abstract = ""
            if isinstance(gold_titles, list) and gold_titles != []:
                gold_title = gold_titles[0]
            else:
                gold_title = ""
            # fill result
            result_tag = et.SubElement(ir, "Result")
            pmid = gold_df.loc[gold_df["id"] == q_id].values[0][1][0]
            result_tag.set("PMID", pmid)
            title = et.SubElement(result_tag, "Title")
            title.text = gold_title
            abstract = et.SubElement(result_tag, "Abstract")
            abstract.text = gold_abstract
        tree = et.ElementTree(root)
        os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
        print(f"Writing gold QA input / IR output to {new_file_name}")
        tree.write(new_file_name, pretty_print=True)
    return new_file_name


def generate_intermediary_datasets(gold_df, qu_output, ir_output):
    # QU: replace concepts with human_concepts and type with gold type
    gold_qu_output = gen_gold_qu_output(gold_df, qu_output)
    # IR: Replace IR with full_abstract in document format
    gold_ir_output = gen_gold_ir_output(gold_df, ir_output)
    return gold_qu_output, gold_ir_output


def get_gold_df(gold_dataset_path):
    with open(gold_dataset_path, "r") as f:
        gold_data = json.loads(f.read())
    # load and flatten data
    gold_df = pd.json_normalize(gold_data, record_path="questions")
    # get gold df
    gold_df["documents"] = gold_df["documents"].apply(get_pmid)
    return gold_df


def run_qu_tests(gold_dataset_path, generation_folder_path, qu_output, tag="gen"):
    gold_df = get_gold_df(gold_dataset_path=gold_dataset_path)
    gen_df = parse_qu_output(qu_output)
    concepts_report = do_concepts_eval(gold_df, gen_df)
    gold_type = gold_df["type"].to_numpy()
    gen_type = gen_df["type"].to_numpy()
    try:
        print(f"{CYAN}Type Evaluation{OFF}")
        print(classification_report(gold_type, gen_type))
        type_report = classification_report(gold_type, gen_type, output_dict=True)
    except:
        type_report = f"TypeReport: Found input variables with inconsistent numbers of samples: [{len(gold_type)}] [{len(gen_type)}]"
    test_results = (
        concepts_report,
        None,
        type_report,
        None,
        None,
        None
    )
    
    save_results(test_results, generation_folder_path, tag)


def run_ir_tests(gold_dataset_path, generation_folder_path, ir_output, tag="gen"):
    gold_df = get_gold_df(gold_dataset_path=gold_dataset_path)
    gen_df = parse_ir_output(ir_output)

    pmids_report = do_pmids_eval(gold_df, gen_df)
    test_results = (None,pmids_report,None,None,None,None)

    save_results(test_results, generation_folder_path, tag)


def run_qa_tests(gold_dataset_path, generation_folder_path, qa_input, tag="gen"):
    factoid_path = generation_folder_path + "/qa/factoid/BioASQform_BioASQ-answer.json"
    list_path = generation_folder_path + "/qa/list/BioASQform_BioASQ-answer.json"
    gold_df = get_gold_df(gold_dataset_path=gold_dataset_path)
    gen_df = parse_qa_output(qa_input, generation_folder_path + "/qa")

    yes_no_report = do_yes_no_eval(gold_df, gen_df)
    factoid_report = do_factoid_eval(gold_df, gen_df, factoid_path)
    list_report = do_list_eval(gold_df, gen_df,list_path)

    test_results = (
        None,
        None,
        None,
        yes_no_report,
        factoid_report,
        list_report,
    )

    save_results(test_results, generation_folder_path, tag)


def run_all_the_tests(gold_dataset_path, generation_folder_path, xml_name, tag="gen"):
    factoid_path = generation_folder_path + "/qa/factoid/BioASQform_BioASQ-answer.json"
    list_path = generation_folder_path + "/qa/list/BioASQform_BioASQ-answer.json"
    generated_qu = generation_folder_path + "/ir/output/" + xml_name

    # get gold df
    gold_df = get_gold_df(gold_dataset_path=gold_dataset_path)
    # get generated df
    gen_df = parse_xml(generated_qu, generation_folder_path + "/qa")

    # do tests
    concepts_report = do_concepts_eval(gold_df, gen_df)
    pmids_report = do_pmids_eval(gold_df, gen_df)
    gold_type = gold_df["type"].to_numpy()
    gen_type = gen_df["type"].to_numpy()
    try:
        print(f"{CYAN}Type Evaluation{OFF}")
        print(classification_report(gold_type, gen_type))
        type_report = classification_report(gold_type, gen_type, output_dict=True)
    except:
        type_report = f"TypeReport: Found input variables with inconsistent numbers of samples: [{len(gold_type)}] [{len(gen_type)}]"
    yes_no_report = do_yes_no_eval(gold_df, gen_df)
    factoid_report = do_factoid_eval(gold_df, gen_df, factoid_path)
    list_report = do_list_eval(gold_df, gen_df,list_path)

    test_results = (
        concepts_report,
        pmids_report,
        type_report,
        yes_no_report,
        factoid_report,
        list_report,
    )
    save_results(test_results, generation_folder_path, tag)
    return test_results


def gen_gold_ir_output_FROM_SNIPPETS(gold_df, gen_folder, xml_name="bioasq_qa.xml"):
    ir_generated = gen_folder + "/ir/output/" + xml_name
    new_file_name = ir_generated.replace(".xml", "_GOLD.xml")

    fileTree = et.parse(ir_generated)
    if fileTree:
        root = fileTree.getroot()
        questions = root.findall("Q")
        for question in questions:
            id = question.attrib.get("id")
            original_question = question.text
            if DEBUG:
                print(original_question)
            ir = question.find("IR")
            # remove original generated articles
            ir.clear()
            snippets = gold_df.loc[gold_df["id"] == id].values[0][6]
            gold_snippet = ""
            for i in range(0, 5):
                try:
                    gold_snippet += snippets[i].get("text")
                except Exception as e:
                    if DEBUG:
                        print(e)
                    gold_snippet += ""

            # gold_snippet =  gold_df.loc[gold_df["id"] == id].values[0][6][0].get("text")
            # gold_abstracts = gold_df.loc[gold_df["id"] == id].values[0][8]
            gold_titles = gold_df.loc[gold_df["id"] == id].values[0][9]
            # system just using top abstract atm
            # if isinstance(gold_abstracts, list):
            #     gold_abstract = gold_abstracts[0]
            # else:
            #     gold_abstract = ""
            if isinstance(gold_titles, list) and gold_titles != []:
                gold_title = gold_titles[0]
            else:
                gold_title = ""
            # fill result
            result_tag = et.SubElement(ir, "Result")
            pmid = gold_df.loc[gold_df["id"] == id].values[0][1][0]
            result_tag.set("PMID", pmid)
            title = et.SubElement(result_tag, "Title")
            title.text = gold_title
            abstract = et.SubElement(result_tag, "Abstract")
            abstract.text = gold_snippet  # pass in the snippet instead of the abstract for those with abstracts
        tree = et.ElementTree(root)
        os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
        print(f"Writing gold QA input / IR output to {new_file_name}")
        tree.write(new_file_name, pretty_print=True)
    return new_file_name


def save_results(test_results, save_path, tag):
    t = time.localtime()
    date = time.strftime("%b-%d-%Y", t)
    (
        concepts_report,
        pmids_report,
        type_report,
        yes_no_report,
        factoid_report,
        list_report,
    ) = test_results

    results_folder = save_path + "/test_results/" + date
    timestamp = time.strftime("%H%M%S", t)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save concepts
    if concepts_report:
        con_name = results_folder + f"/concepts-{timestamp}-{tag}.csv"
        with open(con_name, "w+") as f:
            f1_sum, p_sum, r_sum, scores = concepts_report
            f.write("Average f1 score,Average precision,Average Recall\n")
            f.write(f"{f1_sum},{p_sum},{r_sum}\n")
            f.write("f1,precision,recall\n")
            for f1, p, r in scores:
                f.write(f"{f1},{p},{r}\n")
        print(f"Saved concepts info to {con_name}")

    # Save pmids
    if pmids_report:
        pmid_name = results_folder + f"/pmids-{timestamp}-{tag}.csv"
        with open(pmid_name, "w+") as f:
            f1_sum, p_sum, r_sum, scores = pmids_report
            f.write("Average f1 score,Average precision,Average Recall\n")
            f.write(f"{f1_sum},{p_sum},{r_sum}\n")
            f.write("f1,precision,recall\n")
            for f1, p, r in scores:
                f.write(f"{f1},{p},{r}\n")
        print(f"Saved pmid info to {pmid_name}")

    # Save type report
    if type_report:
        type_name = results_folder + f"/type-{timestamp}-{tag}.json"
        with open(type_name, "w+") as f:
            json.dump(type_report, f, indent=2)
        print(f"Saved type info to {type_name}")

    # Save yes/no
    if yes_no_report:
        yesno_name = results_folder + f"/yesno-{timestamp}-{tag}.csv"
        with open(yesno_name, "w+") as f:
            yf1, yp, yr, nf1, np, nr, f1, p, r = yes_no_report
            f.write("Yes/No f1 score,Yes/No precision ,Yes/No recall\n")
            f.write(f"{f1},{p},{r}\n")
            f.write("Yes f1 score,Yes precision ,Yes recall\n")
            f.write(f"{yf1},{yp},{yr}\n")
            f.write("No f1 score,No precision ,No recall\n")
            f.write(f"{nf1},{np},{nr}\n")
        print(f"Saved yes/no info to {yesno_name}")

    # Save factoids
    if factoid_report:
        fact_name = results_folder + f"/factoid-{timestamp}-{tag}.csv"
        with open(fact_name, "w+") as f:
            lenient_acc, strict_acc, average_mrr, mrrs = factoid_report
            f.write(
                "Factoid average MRR, Factoid strict accuracy, Factoid lenient accuracy,\n"
            )
            f.write(f"{average_mrr},{lenient_acc},{strict_acc}\n")
            f.write("mrr\n")
            for mrr in mrrs:
                f.write(f"{mrr}\n")
        print(f"Saved factoid info to {fact_name}")

    # Save list
    if list_report:
        list_name = results_folder + f"/list-{timestamp}-{tag}.csv"
        with open(list_name, "w+") as f:
            f1_sum, p_sum, r_sum, scores = list_report
            f.write("Average f1 score,Average precision,Average Recall\n")
            f.write(f"{f1_sum},{p_sum},{r_sum}\n")
            f.write("f1,precision,recall\n")
            for f1, p, r in scores:
                f.write(f"{f1},{p},{r}\n")
        print(f"Saved list info to {list_name}")


# Set up the golden answer dataframe
"""
golden_dataset_path = "testing_datasets/augmented_concepts_abstracts_titles.json"
gen_folder = "tmp"
xml_name = "bioasq_qa.xml"

test_results = run_all_the_tests(golden_dataset_path, gen_folder, xml_name)
save_results(test_results, gen_folder)
"""

if __name__ == "__main__":
    print("Running manual analysis")
    golden_dataset_path = "testing_datasets/augmented_concepts_abstracts_titles.json"
    qa_input = "tmp/submit_to_dr_henry/datasets/gen_abs/bioasq_qa_GENERATED_ABSTRACTS.xml"
    qa_output
    gold_df = get_gold_df(golden_dataset_path)
    gen_df = parse_qa_output(qa_input, generation_folder_path + "/qa")
    do_factoid_eval(gold_df, gen_df, gen_factoid_path)

    raw_test_results = analysis.run_qa_tests(
                            gold_dataset_path=golden_dataset_path,
                            generation_folder_path=gen_folder,
                            qa_input=ir_output_generated,
                            tag="gen",
                        )
