"""
This forms the object that encodes a query on the PubMed Index
"""
from typing import List
class PubmedA:
    # seem to be 14,913,938 articles

    def fromDict(self,data: dict):
        pmid = data["pmid"]
        title = data["title"]
        journal = data["journal"]
        mesh_major = data["meshMajor"]
        year = data["year"]
        abstract_text = data["abstractText"]
        return PubmedA(pmid, title, journal,
                             year, abstract_text, mesh_major)

    def __init__(self, pmid: str, title: str, journal: str,
                 year: str, abstract_text: str, mesh_major: List[str]):
        self.journal = journal
        self.mesh_major = mesh_major
        self.year = year
        self.abstract_text = abstract_text
        self.pmid = pmid
        self.title = title

    def __str__(self):
        return f"PMID: {self.pmid}\nTitle: {self.title}\nJournal: {self.journal} | {self.year}\nAbstract Text: {self.abstract_text}\nMESH major: {self.mesh_major}"


