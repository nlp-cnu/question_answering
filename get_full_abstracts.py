import json
import os
import pickle
from lxml import etree as ET

def get_dataset_pmids(gold_dataset,loc_to_save_pmids):
    dataset = None
    with open(gold_dataset) as d:
        dataset = json.loads(d.read())
    questions = dataset["questions"]
    pmids = set()
    for question in questions:
        documents = question.get("documents")
        if documents:
            ids = [url.rsplit('/', 1)[-1] for url in documents]
            if ids:
                pmids.update(ids)
    # save the discovered pmids to file
    print(f"saving {len(pmids)} PMIDs to {loc_to_save_pmids}")
    with open(loc_to_save_pmids,"wb") as f:
        pickle.dump(pmids,f)

def map_abstracts_to_ids(pmids,database_dir,pmid_abstract_path):
    print("Getting the abstract for each pmid")
    with open(pmids,"rb") as f:
        pmids_set = pickle.load(f)
    pmid_abstract_dict = dict()

    #iterate through each db file
    db_files= os.listdir(database_dir)
    for db_file in db_files:
        full_path = database_dir+ "/" + db_file
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
                if id in pmids_set:
                    try:
                        art_ref = article.find('MedlineCitation').find('Article')
                        abstract = art_ref.find('Abstract').find('AbstractText').text
                        title = art_ref.find('ArticleTitle').text
                        pmid_abstract_dict[id] = (title,abstract)
                    except Exception as e:
                        # no abstract for this article
                        print(f"no abstract OR title for PMID {id}")
                        print(str(e))
            except Exception as e:
                print(str(e)) 
    print(f"saving {len(pmid_abstract_dict)} abstracts to {pmid_abstract_path}")
    with open(pmid_abstract_path,"wb") as f:
        pickle.dump(pmid_abstract_dict,f)

def add_full_abstracts(old_dataset, pmid_map, new_dataset):
    print(f"adding full_abstracts to {old_dataset} from {pmid_map} and saving as {new_dataset}")
    with open(pmid_map,"rb") as f:
        pmid_abstract_dict = pickle.load(f)
    dataset = None
    with open(old_dataset) as d:
        dataset = json.loads(d.read())
    questions = dataset["questions"]
    n = 1
    for question in questions:
        documents = question.get("documents")
        print(f"question ({n} / {len(questions)})")
        if documents:
            ids = [url.rsplit('/', 1)[-1] for url in documents]
            title_and_full_abs = [pmid_abstract_dict[id] for id in ids if id in pmid_abstract_dict.keys()]
            titles = [ta[0] for ta in title_and_full_abs]
            full_abs = [ta[1] for ta in title_and_full_abs]
            question["full_abstracts"] = full_abs
            question["titles"] = titles
            print(f"found ({len(full_abs)} / {len(ids)}) full_abstracts for question {n}")
        n=n+1
    with open(new_dataset,'w') as outfile:
        json.dump(dataset,outfile,indent=4)
    print("Done!")

if __name__ == "__main__":
    gold_dataset = "testing_datasets/BioASQ-training8b/training8b.json"
    augmented_dataset = "testing_datasets/BioASQ-training8b/augmented_test_FULL_ABSTRACTS.json"
    pubmed_db_dir = "umls/pubmed"
    pmids_path = "testing_datasets/pmids.txt"
    pmid_abstract_path = "testing_datasets/pmids_and_abstracts.txt"

    get_dataset_pmids(gold_dataset,pmids_path)
    map_abstracts_to_ids(pmids_path,pubmed_db_dir,pmid_abstract_path)
    add_full_abstracts(gold_dataset, pmid_abstract_path, augmented_dataset)
