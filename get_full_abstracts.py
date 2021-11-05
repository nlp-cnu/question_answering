import json
import os
from lxml import etree as ET

# Retrieve the abstracts for each document (same level as pagination)
# IF simple indexing then it is much easier to get abstracts
# IF NOT, then brute force
def get_abstracts_from_db(document_urls,pubmed_db_dir,simple=False):
    # Get the last part of the pubmed document url [[  http://www.ncbi.nlm.nih.gov/pubmed/23959273 -> 23959273  ]]
    ids = [url.rsplit('/', 1)[-1] for url in document_urls]
    full_abstracts = []
    # Open each xml THIS IS VERY BRUTE FORCE AND WILL TAKE A WHILE
    guessed_ids = []
    db_files= os.listdir(pubmed_db_dir)
    for db_file in db_files:
        full_path = pubmed_db_dir+ "/" + db_file
        print(f"opening db: {full_path}")
        pm_tree = ET.parse(full_path)
        if not pm_tree:
            print(f"ERROR: failed to open pubmed file {str(db_file)}")
            return
        root = pm_tree.getroot()
        pubmed_articles = root.findall("PubmedArticle") 
        for article in pubmed_articles:
            try:
                id = article.find("MedlineCitation").find("PMID").text
                if id in ids:
                    try:
                        full_abstracts.append(article.find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text)
                    except Exception as e:
                        # no abstract for this article
                        print(f"no abstract for PMID {id}")
                        print(str(e))
                    guessed_ids.append(id)
                    ids.remove(id)
                if not ids: # when we have found all the abstracts we are looking for
                    return full_abstracts
            except Exception as e:
                print(str(e)) 
    return full_abstracts


# Retrieve the full PubMed abstracts for all articles identified in the golden dataset then augment it
def get_full_abstracts(gold_dataset,augmented_dataset,pubmed_db_dir):
    dataset = None
    with open(gold_dataset) as d:
        dataset = json.loads(d.read())
    questions = dataset["questions"]
    n = 1
    for question in questions:
        documents = question.get("documents")
        print(f"question ({n} / {len(questions)})")
        if documents:
            print(f"num documents: {len(documents)}")
            extracted_abstracts = get_abstracts_from_db(documents,pubmed_db_dir)
            question["full_abstracts"] = extracted_abstracts
            print(f"found {len(extracted_abstracts)} full_abstracts")
        n=n+1
    with open(augmented_dataset,'w') as outfile:
        json.dump(dataset,outfile,indent=4)

if __name__ == "__main__":
    gold_dataset = "testing_datasets/BioASQ-training8b/training8b.json"
    augmented_dataset = "testing_datasets/BioASQ-training8b/augmented_test_FULL_ABSTRACTS.json"

    pubmed_db_dir = "umls/pubmed"
    get_full_abstracts(gold_dataset, augmented_dataset, pubmed_db_dir)