from whoosh import index
from whoosh.fields import Schema, TEXT, IDLIST, ID, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser

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


