# BioASQ-QA-System
> CNU Computer Science capstone project

The BioASQ-QA-System repo provides the source code for a biomedical question answering system aimed at participating in the BioASQ Challenge 8b. I utilized a BioBERT language representation model that was fine-tuned on biomedical data for the purposes of this challenge. I also utilized an index of the PubMed database that is queried by the system for retrieval of relavent biomedical documents related to user questions.


See the [wiki](https://github.com/DanielSims1/BioASQ-QA-System/wiki) for in-depth system documentation or [first time setup guide.](https://github.com/DanielSims1/BioASQ-QA-System/wiki/First-Time-Setup)

## Citation
  Please see [Published BioBERT paper](https://link.springer.com/chapter/10.1007/978-3-030-43887-6_64) for explanation of question answering module pieces: 
```
@InProceedings{10.1007/978-3-030-43887-6_64,
  author="Yoon, Wonjin and Lee, Jinhyuk and Kim, Donghyeon and Jeong, Minbyul and Kang, Jaewoo",
  editor="Cellier, Peggy and Driessens, Kurt",
  title="Pre-trained Language Model for Biomedical Question Answering",
  booktitle="Machine Learning and Knowledge Discovery in Databases",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="727--740",
  isbn="978-3-030-43887-6"
}
```

Here is the original [BioBERT paper](http://dx.doi.org/10.1093/bioinformatics/btz682) since the model used is a BioBERT model.
```
@article{lee2019biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  doi = {10.1093/bioinformatics/btz682}, 
  journal={Bioinformatics},
  year={2019}
}
```
## Requirements
This system can be used in a python 3.7.9 environment with the appropriate dependencies running on a Linux machine.

## License and Disclaimer
Please see and agree to `LICENSE` file for details. Downloading data indicates your acceptance of the disclaimer.

> This work was done in collaboration with Dr. Samuel Henry in preperation for future graduate-level research in the CNU 5-year masters program

