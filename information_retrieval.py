# Based on code from https://github.com/masonnlp/bioasqir

import lxml.etree as ET
import os
from utils import *

import PubmedA

# Here we receive input of the form (id, question, type, entities, query).
# We use this input to query the PubMed database index which has been specially indexed to improve query times.
def search(indexer, parser, query, max_results = 5, batch_mode=False):
    print(f"{MAGENTA}Searching....{OFF}")
    res = []
    if batch_mode:
        q = parser.parse(query)
    else:
        q = parser.parse(query[4])
    with indexer.searcher() as s:
        results = s.search(q, limit=max_results)
        for result in results:
            pa = PubmedA.PubmedA(result.get('pmid'),
                         result.get('title'),
                         result.get('journal'),
                         result.get('year'),
                         result.get('abstract_text'),
                         result.get('mesh_major')) # medical subject headings, keywords
            res.append(pa)
    return res

#Query the the PubMed index with every query generated in the QU module, writing the result articles fetched by query to a file every <write_buffer_size> iterations 
def batch_search(input_file, output_file, indexer, parser, write_buffer_size=500):
    fileTree = ET.parse(input_file)
    if fileTree:
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        root = fileTree.getroot()
        # get all questions from the output file and parse in batch format
        questions = root.findall('Q')
        index = 1
        num_questions = str((len(questions)))
        print(f"{MAGENTA}{num_questions} questions found{OFF}")
        for question in questions:
            # Question ID and question processing tags
            qid = question.get("id")
            qp = question.find("QP")
            # safeguard for malformed query
            if qp.find("Query").text:
                query = qp.find("Query").text
            else:
                print(f"{MAGENTA}No query found, using original question{OFF}")
                query = question.text
            print(f"{MAGENTA}{query} [{index}/{num_questions}]{OFF}")
            # use search method to find a result
            results = search(indexer,parser,query,batch_mode=True)
            if results:
                print(f"{MAGENTA}Results found.{OFF}")
                ir = question.find("IR")
                # create subelements for each result
                for result in results:
                    query_used = ET.SubElement(ir, "QueryUsed")
                    query_used.text = query
                    result_tag = ET.SubElement(ir, "Result")
                    result_tag.set("PMID", result.pmid)
                    journal = ET.SubElement(result_tag, "Journal")
                    journal.text = result.journal
                    year = ET.SubElement(result_tag, "Year")
                    try:
                        year.text = result.year
                    except:
                        pass
                    title = ET.SubElement(result_tag, "Title")
                    title.text = result.title
                    abstract = ET.SubElement(result_tag, "Abstract")
                    abstract.text = result.abstract_text
                    # tags
                    for mesh in result.mesh_major:
                        mesh_major = ET.SubElement(result_tag, "MeSH")
                        mesh_major.text = mesh
                tree = ET.ElementTree(root)
                # save current progress to file every n documents (controlled by write_buffer_size)
                if(index % write_buffer_size-1 == 0):   
                    print(f"{MAGENTA}Writing data to {output_file}{OFF}")
                    tree.write(output_file, pretty_print=True)
            else:
                print(f"{MAGENTA}No results{OFF}")
            index=index+1
        print(f"{MAGENTA}Writing data to {output_file}{OFF}")
        tree.write(output_file, pretty_print=True)
    else:
        print(f"{MAGENTA}Error loading {input_file}{OFF}")


