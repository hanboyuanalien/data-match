import os
import json
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
#load receipts to be matched
df_dataset = pd.read_csv('../新/OCR_compare001.csv',sep=';')
df_datasets=df_dataset.drop(columns=['Unnamed: 0','Unnamed: 3'])
df_datasets['matched_multi_ocr']=''
df_datasets['matched_multi_ocr_all']=''
df_datasets['matched_multi_actual']=''
df_datasets['matched_multi_actual_all']=''
df_datasets.drop_duplicates(keep='first', inplace=True)
#define n-gram
def create_grams(text,n):
    list=[]
    gram=n
    string_grams = ""
    for i in range(0, len(text)-gram+1):
        x=text[i:i+gram]
        list.append(x)
        string_grams = ' '.join(list)
    return string_grams
def creat_bigrams(text):
    return create_grams(text,2)
def creat_trigrams(text):
    return create_grams(text,3)
#load database
df_database = pd.read_csv('../yazio_all.csv')
df_database['item_name_all']=df_database['shop']+df_database['item_name']
df_database=df_database.drop(columns=['Unnamed: 0'])
df_database=df_database.where(df_database.notnull(),None)
df_database['bigram']=df_database['item_name'].apply(creat_bigrams)
df_database['trigram']=df_database['item_name'].apply(creat_trigrams)
df_database['bigram_all']=df_database['item_name_all'].apply(creat_bigrams)
df_database['trigram_all']=df_database['item_name_all'].apply(creat_trigrams)
df_database = df_database[['L1','L2','L3','label','item_name_all','bigram_all','trigram_all','item_name','bigram','trigram','shop','nutrition']]
#set up elasticsearch 
es = Elasticsearch(['http://localhost:9200'],http_auth=['elastic', 'piPh2iel5aat'])
def delete_index() -> bool:
        """
        remove index
        :return: bool
        """
        try:
            es.indices.delete(index='es2', ignore=[400, 404])
            print(f"[INFO] Index 'es2' deleted successfully")
            return True
        except Exception as ex:
            print("[WARNING] some exception has occurred!")
            return False
def index_dataframe(index_name,df):
    delete_index()
    bulk_data = []
    df_dicts = df.to_dict(orient='records')
    for doc in df_dicts:
        action = {"_index": index_name, "_source": doc}
        bulk_data.append(action)
    try:
        helpers.bulk(es, bulk_data, stats_only=False)
        print("dataframe indexed!")
    except Exception as e:
        print(str(e))
index_dataframe('es2',df_database)
#define search strategy
def search(item, search_status):
	search = json.dumps({})
	if search_status == "multi":
		search = json.dumps({
					'query': {
						'bool': {
							'should': [
								{'match': {'bigram': creat_bigrams(item)}},
                                {'match': {'trigram': creat_trigrams(item)}},
                                {'match': {'item_name': item}}
							]
						}
					}
				})
	elif search_status == "multi_all":
		search = json.dumps({
					'query': {
						'bool': {
							'should': [
								{'match': {'bigram_all': creat_bigrams(item)}},
                                {'match': {'trigram_all': creat_trigrams(item)}},
                                {'match': {'item_name_all': item}}
							]
						}
					}
				})
	# to the job.
	result = es.search(index='es2', body=search, size=1)
	if not result['timed_out']:
		data = result['hits']['hits']
		if len(data) > 0:
			digital_items = []
			for item in data:
				digital_items.append(item['_source'])
			return digital_items
	return []
def match_grams(text,n):
    list=[]
    string_grams = ""
    for i in search(text,n):
        x=i['item_name']+';'
        list.append(x)
        string_grams = ' '.join(list)
    return string_grams
def match_grams_all(text,n):
    list=[]
    string_grams = ""
    for i in search(text,n):
        x=i['item_name_all']+';'
        list.append(x)
        string_grams = ' '.join(list)
    return string_grams
def match_multi_all(text):
    return match_grams_all(text,"multi_all")
def match_multi(text):
    return match_grams(text,"multi")
df_datasets['matched_multi_ocr']=df_datasets['OCR'].apply(match_multi)
df_datasets['matched_multi_ocr_all']=df_datasets['OCR'].apply(match_multi_all)
df_datasets['matched_multi_actual']=df_datasets['ACTUAL'].apply(match_multi)
df_datasets['matched_multi_actual_all']=df_datasets['ACTUAL'].apply(match_multi_all)
df_datasets.to_csv('result001.csv')