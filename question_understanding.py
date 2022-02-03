"""
Based on code from https://github.com/masonnlp/bioasq_question_processing
question_understanding.py :
    The Question Understanding (QU) module for the QA pipeline which takes in a query in the form
        [ID, Question]    Example -->  [51406e6223fec90375000009,Does metformin interfere thyroxine absorption?]
    and outputs an xml file containing the original question as well as relevant snippets, features, and a predicted query
    for use in the Information Retrieval portion of the pipeline.
"""
from utils import *
import os
import torch
from lxml import etree as ET

#map the original question to tokens utilizing a tokenizer
def preprocess(df, tokenizer):
    df.encoded_tokens = [tokenizer.encode_plus(text,add_special_tokens=True)['input_ids'] for text in df['Question']] 
    df.attention_mask = [tokenizer.encode_plus(text,add_special_tokens=True)['attention_mask'] for text in df['Question']]
    encoded_tokens = list(df.encoded_tokens)
    attention_mask = list(df.attention_mask)
    
    return encoded_tokens,attention_mask

# Convert indices to Torch tensor and dump into cuda
def feed_generator(device, encoded_tokens,attention_mask):
    batch_size = 16
    batch_seq = [x for x in range(int(len(encoded_tokens)/batch_size))]
    shuffled_encoded_tokens,shuffled_attention_mask = encoded_tokens,attention_mask
    res = len(encoded_tokens)%batch_size
    if res != 0:
        batch_seq = [x for x in range(int(len(encoded_tokens)/batch_size)+1)]
    shuffled_encoded_tokens = shuffled_encoded_tokens+shuffled_encoded_tokens[:res]
    shuffled_attention_mask = shuffled_attention_mask+shuffled_attention_mask[:res]

    for batch in batch_seq:
        maxlen_sent = max([len(i) for i in shuffled_encoded_tokens[batch*batch_size:(batch+1)*batch_size]])
        token_tensor = torch.tensor([tokens+[0]*(maxlen_sent-len(tokens)) for tokens in shuffled_encoded_tokens[batch*batch_size:(batch+1)*batch_size]])
        attention_mask = torch.tensor([tokens+[0]*(maxlen_sent-len(tokens)) for tokens in shuffled_attention_mask[batch*batch_size:(batch+1)*batch_size]])        
        token_tensor = token_tensor.to('cpu')
        attention_mask = attention_mask.to('cpu')
        yield token_tensor,attention_mask

# Returns a prediction ( query, snippets, features)
def predict(device, model,data):
    model.eval()
    if device =="cuda:0":
        model.cuda()
    preds = []
    for token_tensor, attention_mask in data:
        with torch.no_grad():
            logits = model(token_tensor,token_type_ids=None,attention_mask=attention_mask)[0]
            tmp_preds = torch.argmax(logits,-1).detach().cpu().numpy().tolist()
        preds += tmp_preds             
    return preds

# If we are in batch mode, append all generated queries and concepts to xml file,
# Otherwise pass QU data (question type, concepts, query) back for transfer to IR module
def ask_and_receive(questions_df, device, tokenizer, model, nlp , batch_mode = False, output_file=None):
    encoded_tokens_Test,attention_mask_Test = preprocess(questions_df,tokenizer)
    data_test = feed_generator(device, encoded_tokens_Test, attention_mask_Test)
    preds_test = predict(device,model,data_test)
    indices_to_label = {0: 'factoid', 1: 'list', 2: 'summary', 3: 'yesno'}
    predict_label = []
    for i in preds_test[0:len(questions_df['Question'])]:
        for j in indices_to_label:
            if i == j:
                predict_label.append(indices_to_label[j])
    questions_df['type'] = predict_label
    if(batch_mode):
        print(f"{MAGENTA}Writing QU results to xml file...{OFF}")
        xml_tree(questions_df,nlp,output_file)
    else:
        return send_qu_data(questions_df,nlp)

#instead of using the xml, just pass the data
def send_qu_data(df,nlp):
    ind = df.first_valid_index()
    id = df['ID'][ind]
    question = df['Question'][ind]
    type = df['type'][ind]
    doc = nlp(question)
    entities = []
    for ent in doc.ents:
        entities.append(str(ent))
    query = str(' '.join(entities))
    return (id, question, type, entities, query)

# Print the extracted information from BioBERT to an xml file we will append to later.
def xml_tree(df,nlp,output_file):
    root = ET.Element("Input")
    for ind in df.index:
        id = df['ID'][ind]
        question = df['Question'][ind]
        qtype = df['type'][ind]
        q = ET.SubElement(root,"Q")
        q.set('id',str(id))
        q.text = question
        qp = ET.SubElement(q,"QP")
        qp_type = ET.SubElement(qp,'Type')
        qp_type.text = qtype
        doc = nlp(question)
        print(f"{MAGENTA}doc: {doc.ents}{OFF}")
        ent_list = []
        for ent in doc.ents:
            ent_list.append(str(ent))
            qp_en = ET.SubElement(qp,'Entities') 
            qp_en.text = str(ent)
        qp_query = ET.SubElement(qp,'Query')
        qp_query.text = str(' '.join(ent_list))
        # Create IR tag
        IR = ET.SubElement(q, "IR")
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tree.write(output_file, pretty_print=True)
    print(f"writing XML to {output_file}")
