# BioASQ-QA-System
CNU Computer Science capstone project

> NOTE: make sure you have updated pip and python 3.7



# First time setup:
### 1) Navigate to the folder you wish to store this project and clone this repository
```
$ git clone https://github.com/DanielSims1/BioASQ-QA-System.git
```
### 2) Navigate to the cloned repository
```
$ cd BioASQ-QA-System/
``` 

## Ensuring you have the right python version:
### 3) Install `pyenv` for Python version management
  >Installation instructions from [pyenv repository](https://github.com/pyenv/pyenv-installer)
  
  #### 3a) Install: 
  ```
  $ curl https://pyenv.run | bash
  ```
  #### 3b) Restart shell so path changes take effect:
  ```
  $ exec $SHELL
  ```
  
  ### 3c) Follow the steps to have pyenv automatically by adding to ~/.bashrc
  ```
  $ echo 'export PATH="~$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
  ```
  ```
  $ echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  ```
  ```
  $ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
  ```
  ```
  $ exec $SHELL
  ```
  

  ### 4) Install python 3.7.9 with pyenv
  ```
  $ pyenv install 3.7.9
  ```

  ### 5) Set 3.7.9 for local environment
  ```
  $ pyenv local 3.7.9
  ```

## Installing the required dependencies:

  ### 6) Create a python virtual environment `.env` for the project
  ```
  $ python3 -m venv .env
  ```
  ### 7) Activate the `.env` virtual environment
  ```
  $ source .env/bin/activate
  ```
### 8) Use `pip` to install requirements from `requirements.txt`
```
$ pip install -r requirements.txt
```

### 9) Run `qa_system.py`
This will run the system and do any remaining first-time setup such as grabbing the model/index used by the system.
```
$ python qa_system.py
```

> Note you can deactivate the python virtual environment after using the system by simply typing `deactivate`

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
