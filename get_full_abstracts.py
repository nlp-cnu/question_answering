import json
from lxml import etree as ET

# Retrieve the abstracts for each document (same level as pagination)
# IF simple indexing then it is much easier to get abstracts
# IF NOT, then brute force
def get_abstracts_from_db(document_urls,pubmed_articles,simple=False):
    # Get the last part of the pubmed document url [[  http://www.ncbi.nlm.nih.gov/pubmed/23959273 -> 23959273  ]]
    # The articles seem to be indexed on 
    ids = [url.rsplit('/', 1)[-1] for url in document_urls]
    full_abstracts = []
    if simple:
        for id in ids:
            if id < len(pubmed_articles):
                try:
                    full_abstracts.append(pubmed_articles[id].find('Abstract').text)
                except:
                    # no abstract for this article
                    print(f"No Abstract for article {id}")
            else:
                print(f"Article {id} not in PubMed <BY SIMPLE EVALUATION>")
    else:
        guessed_ids = []
        for article in pubmed_articles:
            try:
                id = article.find("PubmedData").find("ArticleIdList")[0].text
                if id in ids:
                    try:
                        full_abstracts.append(pubmed_articles[id].find('Abstract').text)
                    except Exception as e:
                        # no abstract for this article
                        print(str(e))
                    guessed_ids.append(id)
                    ids.remove(id)
                if not ids: # when we have found all the abstracts we are looking for
                    return full_abstracts
            except Exception as e:
                print(str(e)) 
    return full_abstracts


# Retrieve the full PubMed abstracts for all articles identified in the golden dataset then augment it
def get_full_abstracts(gold_dataset,augmented_dataset,pubmed_db, simple=False):
    dataset = None
    with open(gold_dataset) as d:
        dataset = json.loads(d.read())
    pm_tree = ET.parse(pubmed_db)
    if not pm_tree:
        print("ERROR: failed to open pubmed db")
        return
    root = pm_tree.getroot()
    articles = root.findall("PubmedArticle") 
    questions = dataset["questions"]
    n = 1
    for question in questions:
        documents = question.get("documents")
        print(f"({n} / {len(questions)})")
        if documents:
            print(f"num documents: {len(documents)}")
            extracted_abstracts = get_abstracts_from_db(document_urls=documents,pubmed_articles=articles,simple=simple)
            question["full_abstracts"] = extracted_abstracts
        n=n+1
    with open(augmented_dataset,'w') as outfile:
        json.dump(dataset,outfile,indent=4)

if __name__ == "__main__":
    gold_dataset = "testing_datasets/BioASQ-training8b/training8b.json"
    gold_dataset = "testing_datasets/BioASQ-training8b/augmented_test.json"
    augmented_dataset = "testing_datasets/BioASQ-training8b/augmented_test_FULL_ABSTRACTS.json"
    pubmed_db = "umls/pubmed21n0001.xml"
    simple = False
    get_full_abstracts(gold_dataset, augmented_dataset, pubmed_db,simple = simple)