# Based on code from https://github.com/masonnlp/bioasqir

from whoosh import index
from whoosh.fields import Schema, TEXT, IDLIST, ID, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
import lxml.etree as ET

import PubmedA

# here we receive input of the form the form (id, question, type, entities, query). and we want the query
def search(indexer, parser, query, max_results = 5):
    print("Searching....")
    res = []
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

def batch_search(input_file, output_file, indexer, parser, write_buffer_size=1000):
    fileTree = ET.parse(input_file)
    if fileTree:
        root = fileTree.getroot()
        # get all questions from the output file and parse in batch format
        questions = root.findall('Q')
        index = 0
        num_questions = "".join(len(questions))
        for question in questions:
            # Question ID and question processing tags
            qid = question.get('id')
            qp = question.get('QP')
            # safeguard for malformed query
            if qp.find("Query").text:
                query = qp.find("Query").text
            else:
                query = question.text
            print(f"Searching[{index}/{num_questions}]... {query}")
            # use search method to find a result
            results = search(indexer,parser,query)
            if results:
                print("Results found.")
                ir = question.find("IR")
                # create subelements for each result
                for result in results:
                    query_used = ET.SubElement(ir, "QueryUsed")
                    query_used.text = query
                    result = ET.SubElement(ir, "Result")
                    result.set("PMID", result.pmid)
                    journal = ET.SubElement(result, "Journal")
                    journal.text = result.journal
                    year = ET.SubElement(result, "Year")
                    try:
                        year.text = result.year
                    except:
                        pass
                    title = ET.SubElement(result, "Title")
                    title.text = result.title
                    abstract = ET.SubElement(result, "Abstract")
                    abstract.text = result.abstract_text
                    # tags
                    for mesh in result.mesh_major:
                        mesh_major = ET.SubElement(result, "MeSH")
                        mesh_major.text = mesh
                tree = ET.ElementTree(root)
                # save current procress to file every x documents (controlled by write_buffer_size)
                if(index % write_buffer_size == 0):
                    print("Writing data to {output_file}")
                    tree.write(output_file, pretty_print=True)
            index=index+1
        print("Writing data to {output_file}")
        tree.write(output_file, pretty_print=True)
    else:
        print(f"Error loading {input_file}")


