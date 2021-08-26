# This file will download all necessary dependencies, models for the project.
import requests
import re
import os
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    print(f"Downloading file: {os.path.basename(destination)} to {destination}")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    con_length = response.headers.get('Content-Length')
    if con_length:
        total_size = int(response.headers.get('Content-Length'))
    else:
        total_size = float('INF')
    save_response_content(response, destination,total_size)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination,total_size):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        with tqdm(total=total_size,unit_scale=True,unit="B",desc=os.path.basename(destination),initial=0,ascii=True) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

def getModel(cwd):
    print(f"\033[95mRetrieving model\033[0m")
    cwd = cwd + 'model' + os.path.sep
    ensure_file_has_directory(cwd)
    model_files = [('1-0qL0U2LnvdCJb4AYBuO0aGYrlIescjb', cwd + 'config.json'),
                    ('1-1F50xU4Ku-jj-3AzqkfmUJGQ91QKwwu', cwd + 'pytorch_model.bin'),
                    ('1-GaX7s2jHNhStlwVwqFkVIatUAU1LTod', cwd + 'special_tokens_map.json'),
                    ('1-LIJV1BrTWmwIxMjdQsH-zn0GaS9mVBU', cwd + 'tokenizer_config.json'),
                    ('1-3nHfqthCaN3DDu6trbifD1SBz-CAIci', cwd + 'vocab.txt')]
    for file in model_files:
        download_if_nonesistant(file)


def getIndex(cwd):
    print("\033[95mRetrieving index\033[0m")
    cwd = cwd + 'index' + os.path.sep
    ensure_file_has_directory(cwd)
    full = cwd + os.path.sep + 'full_index' + os.path.sep
    partial = cwd + os.path.sep + 'partial_index' + os.path.sep
    ensure_file_has_directory(full)
    ensure_file_has_directory(partial)

    index_files = [('1_35onQtzJf8kFl2EV-HBEAOn2DdAidsX', full + '_pubmed_articles_1.toc'),
                    ('1ZuJAM8niUoTq9ZT28Pp5EJSIDbbNY3aV', full + 'pubmed_articles_liyfs44zssrgfqtn.seg'),
                    ('11Cke43Iqq91CTVvNkQ3w17blRTQ_BNax', full + 'pubmed_articled_WRITELOCK'),
                    ('1wmPeN86iKIAJWye_K0FEk86uZFFetdtR', partial + '_pubmed_articles_1.toc'),
                    ('1AHf9A-oUg7X9HuEUsOvgl0bi2VWoCpqX', partial + 'pubmed_articles_meguiz6sn10his75.seg'),
                    ('1vyuLVhvOue9He7o9-10ejWTlmQ848JP-', partial + '_pubmed_articles_WRITELOCK')]
    for file in index_files:
        download_if_nonesistant(file)

def ensure_file_has_directory(dir):
    if not os.path.isdir(os.path.dirname(dir)):
        print(f'\033[95mMaking {dir} folder\033[0m')
        os.mkdir (dir)

def download_if_nonesistant(file):
    if not os.path.isfile (file[1]) :
        print(f"\033[95m{file[1]} does not exist\033[0m")
        download_file_from_google_drive(file[0],file[1])

#Data Directory is the name of the folder you want your data modules to be stored in
def setup_system(data_directory):
    cwd = os.getcwd()
    dir = cwd + os.path.sep + data_directory + os.path.sep
    ensure_file_has_directory(dir)

    getModel(dir)
    getIndex(dir)

