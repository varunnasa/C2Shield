import subprocess
import pandas as pd
import json
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,Dataset
from torch.autograd import Variable as V
import time
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc
from sklearn.model_selection import train_test_split
import string
import torch.nn as nn
import collections
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
from collections import Counter
import torch.optim as optim
import math

#MLMODEL PREPROCESSING
device='cpu'
class DNSExfiltration(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_out = nn.Linear(128, 1)


        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, input):
        x = self.relu(self.layer_1(input))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        return x


keys = list(string.printable.strip())
print (len(list(string.printable.strip())))

def init(df,param):
    model = DNSExfiltration(98)
    model.load_state_dict(torch.load('./detect_dns_data_exfiltration_using_pretrained_model_in_dsdl.pt',map_location=torch.device('cpu')))
    model = model.to('cpu')
    model.eval()
    return model

text_rows = []
size_avg = []
entropy_avg = []

# this method accepts a dataloader and makes predictions in batches
def predict(dataloader,model):
        predict_label_proba = []
        predict_label = []
        for batch in (dataloader):

            #convert to 1d tensor
            predictions = model(batch.to('cpu'))
            output  = (predictions >= 0.5).int()
            predict_label_proba.extend(predictions)
            predict_label.extend(output)
        predict_label = [x.cpu().detach().numpy().item() for x in predict_label]
        predict_label_proba = [x.cpu().detach().numpy().item() for x in predict_label_proba]
        return predict_label_proba,predict_label

# this method accepts a DNS request and converts into indexes based on printable characters
def index_chars(x):
    request_chars = {}
    # print("-->",x)
    if x:
      for i in range(len(x)):
        try:
            request_chars[keys.index(x[i])] = request_chars.get(keys.index(x[i]), 0) + 1
        except Exception as e:
            continue
      text_rows.append(request_chars)
    else:
      return

#  calculates entropy of a domain
def entropy(domain):
    p, lns = Counter(domain), float(len(domain))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


# removes the subdomain/domain from the request
def replace_tld(x):
    if x is not None:
        return str(x).rsplit('.', 2)[0]
    else:
        return x

# get the subdomain/domain from the request
def get_tld(x):
    without_tld = str(x).rsplit('.', 2)[0]
    return str(x).replace(without_tld,'').lstrip(".")

# compute aggregated features for the same src and subdomain/domain on a window of 10 events
def get_aggregated_features(row,df):
    src = row['src']
    tld = row['tld']
    prev_events = df[(df['src']==src) & (df['tld']==tld)]

    size_avg.append(prev_events['len'].mean())
    entropy_avg.append(prev_events['entropy'].mean())

# prepare input df by calculating features
def prepare_input_df(df):
    keys = list(string.printable.strip())



    df['query'].apply(lambda x: index_chars(x))
    text = pd.DataFrame(text_rows, columns=list(range(0, 94)))
    text.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    text.fillna(0, inplace=True)
    df = pd.concat([text, df], axis=1)


    # request without tld
    df['request_without_domain'] = df['query'].apply(lambda row: replace_tld(row))

    # request without tld
    df['tld'] = df['query'].apply(lambda row: get_tld(row))

    # length of domain
    df['len'] = df['request_without_domain'].apply(len)

    # entropy
    df['entropy'] = df['request_without_domain'].apply(lambda x: entropy(x))

    # take most-recent request
    recent_df = df.loc[df['rank'] == 1]

    # calculate feature by aggregating events
    
    recent_df.apply(lambda x: get_aggregated_features(x,df),axis=1)
    recent_df['size_avg'] = size_avg
    recent_df['entropy_avg'] = entropy_avg
    return recent_df


# apply model on processed dataframe to predict exfiltration
def apply(model,df,param):
    df.drop(['_time'], axis=1,inplace=True, errors='ignore')
    # if recent_df:
    #     del recent_df
    recent_df = prepare_input_df(df)
    input_df = recent_df.drop(['src' ,'query','rank','request_without_domain','tld'], axis=1)
    recent_df.drop(['request_without_domain','tld','len','entropy','size_avg','entropy_avg'], axis=1, inplace=True)
    recent_df.drop(range(0, 94),axis=1,inplace=True)
    input_tensor = torch.FloatTensor(input_df.values)
    dataloader = DataLoader(input_tensor, shuffle=True, batch_size=256)
    predict_is_exfiltration_proba, predict_is_exfiltration = predict(dataloader,model)
    recent_df['pred_is_dns_data_exfiltration_proba'] = predict_is_exfiltration_proba
    recent_df['pred_is_dns_data_exfiltration'] = predict_is_exfiltration
    # print(recent_df.columns)
    # print(df.columns)
    text_rows.clear()
    size_avg.clear()
    entropy_avg.clear()
    output = pd.merge(recent_df,df,on=['src','query','rank'],how='right')
    return output

# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = DNSExfiltration(98)
    model.load_state_dict(torch.load('./detect_dns_data_exfiltration_using_pretrained_model_in_dsdl.pt',map_location=torch.device('cpu')))
    model = model.to('cpu')
    model.eval()
    return model
# ############################################################
def extract_zeek_data(zeek_log_file):
    # Assuming the format of the Zeek log file
    # Modify accordingly if the format is different
    fields_to_extract = ["id.orig_h", "query", "ts"]
    # 2, 9, 0

    # Read the Zeek log file
    with open(zeek_log_file, 'r') as zeek_file:
        lines = zeek_file.readlines()

    # Parse each line to extract required fields
    data = []
    for line in lines:
        # Split the line by tabs
        fields = line.strip().split('\t')
        if len(fields)>= 23 and (fields[0]!="#fields" and fields[0]!="#types"):
            # print(fields[0])
            # Extract required fields
            extracted_fields = [fields[2],fields[9],float(fields[0])]
            data.append(extracted_fields)

    # Create DataFrame from the extracted data
    df = pd.DataFrame(data, columns=["src", "query", "_time"])

    # Exclude rows where query is None or empty
    df = df.dropna(subset=['query']).loc[df['query'] != '']

    if not df.empty:
        # Convert _time to numeric
        df['_time'] = pd.to_numeric(df['_time'])
        # Sort DataFrame by src, query, and _time
        df.sort_values(by=['src', 'query', '_time'], inplace=True)
        # Add rank column
        df['rank'] = df.groupby(['src', 'query'])['_time'].rank(method='min')

    return df
# ############################################################

def extract_pcap_data(pcap_file):
    # Run tshark command to extract required fields
    tshark_cmd = [
        "tshark",
        "-r", pcap_file,
        "-T", "fields",
        "-e", "ip.src",
        "-e", "dns.qry.name",
        "-e", "frame.time_epoch"
    ]
    result = subprocess.run(tshark_cmd, capture_output=True, text=True)
    
    # Check if tshark command was successful
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        data = [line.split('\t') for line in lines]
        df = pd.DataFrame(data, columns=["src","query","_time"])  # Modified column names
        # Exclude rows where query is None or empty
        df = df.dropna(subset=['query']).loc[df['query'] != '']
        if not df.empty:
            df.sort_values(by=['src', 'query', '_time'], inplace=True)
            df['rank'] = df.groupby(['src', 'query'])['_time'].rank(method='min')
        return df
    else:
        print("Error running tshark command.")
        return None

# Prompt user to input the path of the pcap file
# pcap_file = '/content/drive/MyDrive/botnet-capture-20110810-neris.pcap'
# df = extract_pcap_data(pcap_file)

# if df is not None:
#     print("Data extracted successfully:")
#     print(df.head())
#     model = load("model")
#     recent_df = apply(model,df,"")
#     print(recent_df)
# else:
#     print("Failed to extract data.")
