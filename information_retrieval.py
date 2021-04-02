# Based on code from https://github.com/masonnlp/bioasqir

from whoosh import index
from whoosh.fields import Schema, TEXT, IDLIST, ID, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
import lxml.etree as ET

import PubmedA

# here we receive input of the form the form (id, question, type, entities, query). and we want the query
def search(indexer, parser, query, max_results = 5, batch_mode=False):
    print("Searching....")
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

def batch_search(input_file, output_file, indexer, parser, write_buffer_size=500):
    fileTree = ET.parse(input_file)
    if fileTree:
        root = fileTree.getroot()
        # get all questions from the output file and parse in batch format
        questions = root.findall('Q')
        index = 1
        num_questions = str((len(questions)))
        print(f"{num_questions} questions found")
        for question in questions:
            # Question ID and question processing tags
            #print(question.text)
            qid = question.get("id")
            qp = question.find("QP")
            # safeguard for malformed query
            if qp.find("Query").text:
                query = qp.find("Query").text
            else:
                print("No query found, using original question")
                query = question.text
            print(f"{query} [{index}/{num_questions}]")
            # use search method to find a result
            results = search(indexer,parser,query,batch_mode=True)
            if results:
                print("Results found.")
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
                # save current procress to file every x documents (controlled by write_buffer_size)
                if(index % write_buffer_size-1 == 0):   
                    print(f"Writing data to {output_file}")
                    tree.write(output_file, pretty_print=True)
            else:
                print("No results")
            
            index=index+1
        print(f"Writing data to {output_file}")
        tree.write(output_file, pretty_print=True)
    else:
        print(f"Error loading {input_file}")


