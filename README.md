# BioASQ-QA-System
CNU Computer Science capstone project


## First time setup:
### 1) Navigate to the folder you wish to store this project and clone this repository
```
git clone https://github.com/DanielSims1/BioASQ-QA-System.git
```
### 2) Navigate to the cloned repository
```
cd BioASQ-QA-System
``` 

### 3) Use `pip` or `pip3` to install requirements from `requirements.txt`
> For `pip`
```
pip install -r requirements.txt
```
> For `pip3`
```
pip3 install -r requirements.txt
```
### 4) Run `qa_system.py` with `python 3`
This will run the system and do any remaining first-time setup such as grabbing the model/index used by the system.
```
python3 qa_system.py
```


# Citation
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
See [Wiki for documentation and setup guide](https://github.com/DanielSims1/BioASQ-QA-System/wiki)
