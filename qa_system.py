"""
This is the main file for the QA pipeline. 
    It utilizes question_understanding.py to extract type, concepts, and a query from a plaintext question.
    It then utilizes information_retrieval.py to retrieve a list of PubMed articles pertaining to the query formed previously.
    Finally it utilizes question_answering.py to generate an answer to the original question utilizing the information gathered in the previous two steps.
"""
from utils import *

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import spacy
import en_core_sci_lg
import os
import shutil
import argparse

from whoosh import index
from whoosh.fields import Schema, TEXT, IDLIST, ID, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser

import setup
import question_understanding
import information_retrieval2
import question_answering
import analysis

if __name__ == "__main__":
    # Define args for system
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-E",
        "--Evaluation",
        dest="evaluate",
        action="store_true",
        help="Run system in evaluation mode, requires a file input defined by -i <file_name>",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        help="This defines the input file for the system. A question list for QU, an xml with queries and questions for IR, or an xml with questions, question types, and expert texts for QA.",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Shows more verbose output for system",
    )
    args = parser.parse_args()
    # Used for logging
    DEBUG = args.verbose
    if args.evaluate and not args.input:
        raise argparse.ArgumentError(
            "You must define an input file with qa_system.py -E -i <file_name> if you want to run evaluations"
        )
    # This ensures that all the packages are installed so that the system can work with the modules
    data_folder = "data_modules"
    setup.setup_system(data_folder)
    index_folder_name = "index"
    model_folder_name = "model"
    pubmed_official_index_name = "pubmed_articles"
    if args.evaluate:
        print(f"System is in EVAL mode <{GREEN}{args.evaluate}{OFF}>")

        while True:
            # NORMAL VALUES
            qu_input = "testing_datasets/input.csv"
            ir_input_generated = "tmp/ir/input/bioasq_qa.xml"
            ir_output_generated = "tmp/ir/output/bioasq_qa.xml"
            qa_output_generated_dir = "tmp/qa/"

            # For evaluation
            # qu_input = "testing_datasets/evaluation_input.csv"
            # ir_input_generated = "tmp/ir/input/bioasq_qa_EVAL.xml"
            # ir_output_generated = "tmp/ir/output/bioasq_qa_EVAL.xml"
            # qa_output_generated_dir = "tmp/qa_EVAL/"

            # SMALL BATCH
            # qu_input = "tmp/small_batch/qu/input/small_input.csv"
            # ir_input_generated = "tmp/small_batch/ir/input/bioasq_qa_SMALL.xml"
            # ir_output_generated = "tmp/small_batch/ir/output/bioasq_qa_SMALL.xml"
            # qa_output_generated_dir = "tmp/small_batch/qa/"

            # User prompt
            batch_options = f"""{MAGENTA}
            What part of the system do you want to evaluate? (Any non-number input will Cancel) 
            0) Whole system
            1) Question Understanding (QU)
            2) Information Retrieval (IR)
            3) Question Answering (QA)
            4) QU + IR
            5) IR + QA{OFF}
            """
            eval_options_dict = {
                "0": "Whole system",
                "1": "Question Understanding",
                "2": "Information Retrieval",
                "3": "Question Answering",
                "4": "QU + IR",
                "5": "IR + QA",
            }
            result = input(batch_options)
            if result:
                golden_dataset_path = (
                    "testing_datasets/augmented_concepts_abstracts_titles.json"
                )
                # folder where intermediary datasets are stored
                gen_folder = "tmp"
                xml_name = os.path.basename(
                    args.input
                )  # set the name for storing the intermediary datasets
                # xml_name = "bioasq_qa.xml"

                if result in eval_options_dict.keys():
                    print(f"{MAGENTA}{eval_options_dict.get(result)} selected.{OFF}")

                # do setup for modules using QU
                if result in ["0","1","4"]:
                    # This is for cpu support for non-NVIDEA cuda-capable machines.
                    spacy.prefer_gpu()
                    # initialize model
                    print(f"{MAGENTA}Initializing model...{OFF}")
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    model = BertForSequenceClassification.from_pretrained(
                        data_folder + os.path.sep + model_folder_name, cache_dir=None
                    )
                    # load in BioBERT
                    print(f"{MAGENTA}Loading BioBERT...{OFF}")
                    nlp = en_core_sci_lg.load()
                # do setup for IR
                if result in ["0","2", "4", "5"]:
                    # load index
                    index_var = "full_index"
                    print(f"{MAGENTA}Loading index...{OFF}")
                    # This is the schema for each query retrieved from Pubmed
                    pubmed_article_ix = index.open_dir(
                        data_folder + os.path.sep + index_folder_name + os.path.sep + index_var,
                        indexname=pubmed_official_index_name,
                    )
                    qp = QueryParser(
                        "abstract_text",
                        schema=Schema(
                            pmid=ID(stored=True),
                            title=TEXT(stored=True),
                            journal=TEXT(stored=True),
                            mesh_major=IDLIST(stored=True),
                            year=NUMERIC(stored=True),
                            abstract_text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                        ),
                    )
                
                if result == "0":
                    # Run Full System
                    print("Running full system")
                    # use the question list from user
                    qu_input = args.input

                    questions_df = pd.read_csv(qu_input, sep=",", header=0)
                    question_understanding.ask_and_receive(
                        questions_df=questions_df,
                        device=device,
                        tokenizer=tokenizer,
                        model=model,
                        nlp=nlp,
                        batch_mode=True,
                        output_file=ir_input_generated,
                    )
                    information_retrieval2.batch_search(
                        input_file=ir_input_generated,
                        output_file=ir_output_generated,
                        indexer=pubmed_article_ix,
                        parser=qp,
                    )
                    question_answering.run_batch_mode(
                        input_file=ir_output_generated,
                        output_dir=qa_output_generated_dir,
                    )
                    # Run tests
                    print("Running tests")
                    raw_test_results = analysis.run_all_the_tests(
                        golden_dataset_path, gen_folder, xml_name, "gen"
                    )
                    # Convert QU output / IR input to gold
                    print("Convert QU output / IR input to gold")
                    gold_df = analysis.get_gold_df(golden_dataset_path)
                    gold_ir_input = analysis.gen_gold_qu_output(
                        gold_df=gold_df, gen_folder=gen_folder, xml_name=xml_name
                    )
                    # Run IR and QA
                    print("Run IR and QA")
                    information_retrieval2.batch_search(
                        input_file=gold_ir_input,
                        output_file=ir_output_generated,
                        indexer=pubmed_article_ix,
                        parser=qp,
                    )
                    question_answering.run_batch_mode(
                        input_file=ir_output_generated,
                        output_dir=qa_output_generated_dir,
                    )
                    # Run tests
                    print("Run tests with gold QU output")
                    gold_qu_test_results = analysis.run_all_the_tests(
                        gold_dataset_path=golden_dataset_path,
                        generation_folder_path=gen_folder,
                        xml_name=xml_name,
                        tag="mixed",
                    )
                    # Convert IR output
                    print("Convert IR output")
                    gold_ir_output = analysis.gen_gold_ir_output_FROM_SNIPPETS(
                        gold_df=gold_df, gen_folder=gen_folder, xml_name=xml_name
                    )
                    # the new naming scheme is based off the new filename with _GOLD
                    xml_name = os.path.basename(gold_ir_output)

                    # Run QA
                    print("Run QA")
                    question_answering.run_batch_mode(
                        input_file=gold_ir_output, output_dir=qa_output_generated_dir
                    )
                    # Run tests
                    print("Run tests with all gold input")
                    gold_qu_ir_test_results = analysis.run_all_the_tests(
                        gold_dataset_path=golden_dataset_path,
                        gen_folder=gen_folder,
                        xml_name=xml_name,
                        tag="-top-5-gold-SNIPPETS",
                    )
                    # Finished with system analysis
                    print(f"{CYAN}Finished full system analysis!{OFF}")

                elif result == "1":
                    qu_input = args.input
                    questions_df = pd.read_csv(qu_input, sep=",", header=0)
                    # run with user input
                    question_understanding.ask_and_receive(
                        questions_df=questions_df,
                        device=device,
                        tokenizer=tokenizer,
                        model=model,
                        nlp=nlp,
                        batch_mode=True,
                        output_file=ir_input_generated,
                    )
                    raw_test_results = analysis.run_qu_tests(
                        gold_dataset_path=golden_dataset_path,
                        generation_folder_path=gen_folder,
                        qu_output=ir_input_generated,
                        tag="gen",
                    )
                    # compare to golden output
                    # gold_df = analysis.get_gold_df(golden_dataset_path)
                    # gold_ir_input = analysis.gen_gold_qu_output(
                    #     gold_df=gold_df, gen_folder=gen_folder, xml_name=xml_name
                    # )
                    # xml_name = os.path.basename(gold_ir_input)
                    # raw_test_results = analysis.run_qu_tests(
                    #     gold_dataset_path=golden_dataset_path,
                    #     generation_folder_path=gen_folder,
                    #     qu_output=ir_input_generated,
                    #     tag="gold",
                    # )

                elif result == "2":
                    ir_input_generated = args.input
                    if os.path.exists(ir_input_generated):
                        information_retrieval2.batch_search(
                            input_file=ir_input_generated,
                            output_file=ir_output_generated,
                            indexer=pubmed_article_ix,
                            parser=qp,
                        )
                        raw_test_results = analysis.run_ir_tests(
                            gold_dataset_path=golden_dataset_path,
                            generation_folder_path=gen_folder,
                            ir_output=ir_output_generated,
                            tag="gen",
                        )

                        # compare to golden output
                        # gold_df = analysis.get_gold_df(golden_dataset_path)
                        # gold_ir_input = analysis.gen_gold_ir_output(
                        #     gold_df=gold_df, gen_folder=gen_folder, xml_name=xml_name
                        # )
                        # xml_name = os.path.basename(gold_ir_input)
                        # raw_test_results = analysis.run_ir_tests(
                        #     gold_dataset_path=golden_dataset_path,
                        #     generation_folder_path=gen_folder,
                        #     ir_output=ir_output_generated,
                        #     tag="gold",
                        # )
                    else:
                        print(
                            f"{RED}Make sure you run the QU module before running the IR module.{OFF}"
                        )
                elif result == "3":
                    ir_output_generated = args.input
                    if os.path.exists(ir_output_generated):
                        question_answering.run_batch_mode(
                            input_file=ir_output_generated,
                            output_dir=qa_output_generated_dir,
                        )

                        raw_test_results = analysis.run_qa_tests(
                            gold_dataset_path=golden_dataset_path,
                            generation_folder_path=gen_folder,
                            qa_input=ir_output_generated,
                            tag="gen",
                        )
                        # compare to gold
                    else:
                        print(
                            f"{RED}Make sure you run both the QU module and the IR module before running the QA module.{OFF}"
                        )
                elif result == "4":
                    qu_input = args.input
                    questions_df = pd.read_csv(qu_input, sep=",", header=0)
                    # run with user input
                    question_understanding.ask_and_receive(
                        questions_df=questions_df,
                        device=device,
                        tokenizer=tokenizer,
                        model=model,
                        nlp=nlp,
                        batch_mode=True,
                        output_file=ir_input_generated,
                    )
                    raw_test_results = analysis.run_qu_tests(
                        gold_dataset_path=golden_dataset_path,
                        generation_folder_path=gen_folder,
                        qu_output=ir_input_generated,
                        tag="gen",
                    )
                    # compare to golden output
                    # gold_df = analysis.get_gold_df(golden_dataset_path)
                    # gold_ir_input = analysis.gen_gold_qu_output(
                    #     gold_df=gold_df, gen_folder=gen_folder, xml_name=xml_name
                    # )
                    # xml_name = os.path.basename(gold_ir_input)
                    # raw_test_results = analysis.run_qu_tests(
                    #     gold_dataset_path=golden_dataset_path,
                    #     generation_folder_path=gen_folder,
                    #     qu_output=ir_input_generated,
                    #     tag="gold",
                    # )
                    information_retrieval2.batch_search(
                        input_file=ir_input_generated,
                        output_file=ir_output_generated,
                        indexer=pubmed_article_ix,
                        parser=qp,
                    )

                    raw_test_results = analysis.run_ir_tests(
                        gold_dataset_path=golden_dataset_path,
                        generation_folder_path=gen_folder,
                        ir_output=ir_output_generated,
                        tag="gen",
                    )

                    # compare to golden output
                    # gold_df = analysis.get_gold_df(golden_dataset_path)
                    # gold_ir_input = analysis.gen_gold_ir_output(
                    #     gold_df=gold_df, gen_folder=gen_folder, xml_name=xml_name
                    # )
                    # xml_name = os.path.basename(gold_ir_input)
                    # raw_test_results = analysis.run_ir_tests(
                    #     gold_dataset_path=golden_dataset_path,
                    #     generation_folder_path=gen_folder,
                    #     ir_output=ir_output_generated,
                    #     tag="gold",
                    # )
                elif result == "5":
                    ir_input_generated = args.input
                    if os.path.exists(ir_input_generated):
                        information_retrieval2.batch_search(
                            input_file=ir_input_generated,
                            output_file=ir_output_generated,
                            indexer=pubmed_article_ix,
                            parser=qp,
                        )
                        raw_test_results = analysis.run_ir_tests(
                            gold_dataset_path=golden_dataset_path,
                            generation_folder_path=gen_folder,
                            ir_output=ir_output_generated,
                            tag="gen",
                        )

                        # # compare to golden output
                        # gold_df = analysis.get_gold_df(golden_dataset_path)
                        # gold_ir_input = analysis.gen_gold_ir_output(
                        #     gold_df=gold_df, gen_folder=gen_folder, xml_name=xml_name
                        # )
                        # xml_name = os.path.basename(gold_ir_input)
                        # raw_test_results = analysis.run_ir_tests(
                        #     gold_dataset_path=golden_dataset_path,
                        #     generation_folder_path=gen_folder,
                        #     ir_output=ir_output_generated,
                        #     tag="gold",
                        # )

                        question_answering.run_batch_mode(
                            input_file=ir_output_generated,
                            output_dir=qa_output_generated_dir,
                        )

                        raw_test_results = analysis.run_qa_tests(
                            gold_dataset_path=golden_dataset_path,
                            generation_folder_path=gen_folder,
                            qa_input=ir_output_generated,
                            tag="gen",
                        )

                    else:
                        print(
                            f"{RED}Make sure you run the QU module before running the IR module.{OFF}"
                        )
                        
                else:
                    print(f"{MAGENTA}Shutting down...{OFF}")
                    quit()
    # If the user responds with anything not affirmative, send them to the live question answering
    else:
        #LOAD ALL THE MODELS

        # This is for cpu support for non-NVIDEA cuda-capable machines.
        spacy.prefer_gpu()
        # initialize model
        print(f"{MAGENTA}Initializing model...{OFF}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            data_folder + os.path.sep + model_folder_name, cache_dir=None
        )
        # load in BioBERT
        print(f"{MAGENTA}Loading BioBERT...{OFF}")
        nlp = en_core_sci_lg.load()
        # load index
        index_var = "full_index"
        print(f"{MAGENTA}Loading index...{OFF}")
        # This is the schema for each query retrieved from Pubmed
        pubmed_article_ix = index.open_dir(
            data_folder + os.path.sep + index_folder_name + os.path.sep + index_var,
            indexname=pubmed_official_index_name,
        )
        qp = QueryParser(
            "abstract_text",
            schema=Schema(
                pmid=ID(stored=True),
                title=TEXT(stored=True),
                journal=TEXT(stored=True),
                mesh_major=IDLIST(stored=True),
                year=NUMERIC(stored=True),
                abstract_text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            ),
        )
        n = 0
        while True:
            user_question = input(
                f"{MAGENTA}:: Please enter your question for the BioASQ QA system or 'quit' ::\n{OFF}"
            )
            # handle end loop
            if user_question == "quit":
                quit()
            df = pd.DataFrame({"ID": [n], "Question": user_question})
            # Retrieve the id,type, concepts, and query generated by QU module
            qu_output = question_understanding.ask_and_receive(
                questions_df=df,
                device=device,
                tokenizer=tokenizer,
                model=model,
                nlp=nlp,
            )
            id, question, type, concepts, query = qu_output
            if type == "summary":
                print(
                    f"{RED}Summary type questions are currently not supported. \nPlease try asking a question that can be answered with a list, yes/no, or factoid style answer.{OFF}"
                )
            else:
                print(
                    f"{MAGENTA} <QU>\nID: {id}\nQuestion: {question}\nType: {type}\nConcepts:{concepts}\nQuery: {query}\n</QU> {OFF}"
                )
                query_results = information_retrieval2.search(
                    pubmed_article_ix, qp, qu_output
                )
                if query_results:
                    top_result = query_results[0]
                    print(f"{MAGENTA} Top result\n{top_result}{OFF}")
                    # Pass in the question ID, type, user question, and top abstract for the result
                    data_for_qa = (n, type, user_question, top_result.abstract_text)
                    # all temporary data will be stored in tmp/live_qa/
                    qa_output_generated_dir = f"{os.getcwd()}{os.path.sep}tmp{os.path.sep}live_qa{os.path.sep}"
                    results = question_answering.get_answer(
                        data_for_qa, output_dir=qa_output_generated_dir
                    )
                    if results:
                        if type == "list":
                            # get the first key
                            index = list(results.keys())[0]
                            top_three_answers = f"1) {results[index][0]['text']}\n\t 2) {results[index][1]['text']}\n\t 3) {results[index][2]['text']}"
                            results = top_three_answers  ## give more answers for list-style questions
                        print(
                            f"{YELLOW}****************************************************************************************{OFF}\n\n \033[92m [<QUESTION>]{OFF}\n\t{MAGENTA}'{user_question}' {OFF} \n  \033[92m[<ANSWER>]{OFF}\n\t{MAGENTA} {results} {OFF} \n\n{YELLOW}****************************************************************************************{OFF}"
                        )
                        # Cleaning up all generated temp files
                        # clear_tmp_dir("tmp")
                    else:
                        print(
                            f"{RED}The Question Answering model encountered an error when trying to answer your question.{OFF}"
                        )
                else:
                    print(
                        f"{RED}Unfortunately I do not know the answer to your question.{OFF}"
                    )
            n += 1


def clear_tmp_dir(dir):
    print(f"{WHITE}tmp dir cleaning{OFF}")
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
