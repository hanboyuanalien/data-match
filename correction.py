from itertools import islice
from symspellpy import SymSpell, Verbosity
from somajo import SoMaJo
tokenizer = SoMaJo("de_CMC", split_camel_case=True,split_sentences=True)
def measurement(o):
    s=[]
    m=[]
    sentences = tokenizer.tokenize_text(o)
    for i, sentence in enumerate(sentences):
        for token in sentence:
            if token.token_class=='measurement' or token.token_class=='number':
                m.append(token.text)
                
            elif token.text!=['',',']:
                s.append(token.text)
    return s 


def abbre_extract(o):
    abbre=[]
    for a in o:
        if re.search(r'\.',a):
            abbre.append(a)     
    return abbre

def shop(o):
    shop=[]
    abbr=[]
    r=[]
    abbre=[]
    r1=[]
    for a in o:
        sentences = re.split(r"([.,])", a)
        sentences.append("")
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        abbre = abbre+sentences
    if len(abbre[0])<4:
        if abbre[0].isupper() or re.search(r'\d|\-|\&',abbre[0]):
            shop.append(abbre[0])
        elif re.search(r'\.',abbre[0]):
            abbr.append(abbre[0])  
        else:r.append(abbre[0])
    elif re.search(r'\.',abbre[0]):
        abbr.append(abbre[0])  
    else:r.append(abbre[0])
    for a in abbre[1:]:
        if a=='0':
            a=='O'
        if re.search(r'\.',a):
            abbr.append(a) 
        else:r.append(a)
    #if len(r)>0 and len(r[0])<4 and r[0].isupper():
        #shop.append(r[0])
    #if len(r)>0 and len(r[0])<4 and re.search(r'\d|\-|\&',r[0]):
        #shop.append(r[0])
    return shop
def percen(o):
    x=[]
    for i,b in enumerate(o):
        if re.search(r'\%',b):
            if len(b)<3 and re.search(r'\d',o[i-1]):
                x.append(''.join(''.join(re.findall(r'\d|\,',o[i-1]))+b))
            elif re.search(r'\%',b):
                x.append(b)
    return x
def combine(o):
    shop=[]
    abbr=[]
    r=[]
    abbre=[]
    for a in o:
        sentences = re.split(r"([.,])", a)
        sentences.append("")
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        abbre = abbre+sentences
    if len(abbre[0])<4:
        if abbre[0].isupper() or re.search(r'\d|\-|\&',abbre[0]):
            shop.append(abbre[0])  
        else:r.append(abbre[0])  
    else:r.append(abbre[0])
    for a in abbre[1:]:
        if a != '':
            if a=='0':
                r.append('o.')
            if a=='01':
                r.append('öl')
            else:r.append(a)
    s=[]
    m=[]
    for x in r:
        if re.search(r'\.',x):
            s.append(x)
        else:
            tokens = tokenizer.tokenize_text([x])
            for i, sentence in enumerate(tokens):
                for token in sentence: 
                    if token.token_class=='measurement' or token.token_class=='number':
                        m.append(token.text)
                
                    elif token.text!=['',',']:
                        s.append(token.text)
    u=[]
    q=[]
    for x in s:
        if re.search(r'\-',x):
            u.append(x.split('-')[0])
            u.append(x.split('-')[1])
        else:u.append(x)
    for y in u:
        if y not in ['',',','%','3.']:
            q.append(y)
    return q

def singlecorrection(o):
    sym_spell = SymSpell(max_dictionary_edit_distance=5)
    dictionary_path = 'new.txt'
    #dictionary_path = 'word_combine_lower.txt'
    #dictionary_path = 'word_supervised.txt'
    #dictionary_path = 'word_withshop.txt'
    sym_spell.load_dictionary(dictionary_path, 0, 1, separator="$")
    w=[]
    for a in o:
        suggestions = sym_spell.lookup(a.lower().replace('10','lo').replace('1','i'), Verbosity.CLOSEST, max_edit_distance=5)
        if suggestions:
            w.append(suggestions[0].term)
        else: w.append(a)
    return w

def segcorrection(o):
    sym_spell = SymSpell(max_dictionary_edit_distance=5)
    dictionary_path = 'new.txt'
    #dictionary_path = 'word_combine_lower.txt'
    #dictionary_path = 'word_supervised.txt'
    #dictionary_path = 'word_withshop.txt'
    sym_spell.load_dictionary(dictionary_path, 0, 1, separator="$")
    w=[]
    a=''.join(o).replace('10','lo').replace('1','i')
    if a:
        result=sym_spell.word_segmentation(a.lower(),max_edit_distance=5)
        w.append(result.corrected_string)
    else: w.append(a)
    return w

import os
import re
import pandas as pd
stop_word=['UKE','uke','vak.','VKE','QS','Stück','kg','-QS','ST','x','IL','Kg','US','OKT','10ST','VkE','-ST','10er','6er','P','BDH', '10ST', '0GT', '10ER']
df = pd.read_csv('../新/OCR_compare001.csv',sep=';')
df =df.drop(columns=['Unnamed: 0','Unnamed: 3'])
df.drop_duplicates(keep='first', inplace=True)
df['edit']=df['OCR'].apply(lambda x: [word for word in re.split(' |/',x)if word not in stop_word])
df['percentage']=df['edit'].apply(percen)
df['shop']=df['edit'].apply(shop)
df.head(200)
df['combine']=df['edit'].apply(combine)
df['singlecorrection']=df['combine'].apply(singlecorrection)
df['segcorrection']=df['combine'].apply(segcorrection)
df.head(200)

import numpy as np
import Levenshtein 
import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt
c2v_model = chars2vec.load_model('eng_150')
def score(arg1, arg2, arg3):
    texta =''.join(arg1).lower()
    textb = ''.join(arg2)
    textc = ''.join(''.join(arg3).split(" "))
    if textb!=textc:
        words = [texta, textb, textc]
    else:
        return 1
    textc1=[]
    for x in ''.join(arg3).split(" "):
        textc1.append(x)
    
    a= len(arg1)-len(arg2)
    b= len(arg1)-len(textc1)
    if a<0 or b<0:
        if a>b:
            return arg2
        else:
            return textc1
    else:
        score1=Levenshtein.distance(texta,textb)
        score2=Levenshtein.distance(texta,textc)
        if score2 < score1:
            return ''.join(arg3).split(" ")
        else:
            return arg2
        
    #if texta!='':
        #word_embeddings = c2v_model.vectorize_words(words)
        #score1="%10.4f" % np.linalg.norm(word_embeddings[0] - word_embeddings[1])
        #score2="%10.4f" % np.linalg.norm(word_embeddings[0] - word_embeddings[2])
    #score1=Levenshtein.distance(texta,textb)
    #score2=Levenshtein.distance(texta,textc)
    #if score2 < score1:
        #return ''.join(arg3).split(" ")
    #else:
        #return arg2
    
def func(arg1, arg2):
    if ''.join(arg1)==''.join(''.join(arg2).split(" ")):
        return arg1
df['same_result']= df.apply(lambda row:func(row['singlecorrection'],row['segcorrection']), axis = 1)
df['final']=df.apply(lambda row:score(row['combine'],row['singlecorrection'],row['segcorrection']), axis = 1)
df

import os.path
import collections
from operator import itemgetter

WORDFILE = 'new_CORRECT1_1.txt'
class Autocorrect(object):
    def __init__(self, ngram_size=3, len_variance=3):
        self.ngram_size = ngram_size
        self.len_variance = len_variance
        self.words = set([w.lower().strip() for w in open(WORDFILE).read().splitlines()])
        self.ngram_words = collections.defaultdict(set)
        for word in self.words:
            for ngram in self.ngrams(word):
                self.ngram_words[ngram].add(word)
    def lookup(self, word):
        return word.lower() in self.words
    def ngrams(self, word):
        all_ngrams = set()
        for i in range(0, len(word) - self.ngram_size + 1):
            all_ngrams.add(word[i:i + self.ngram_size])
        return all_ngrams
    def change(o):
        a=''
        for x in o:
            a=a+" "+x
        return ''.join(a)

    def suggested_words(self, target_word, results=5):
        word_ranking = collections.defaultdict(int)
        possible_words = set()
        if target_word != 1:
            target_word=change(target_word)[1:]
        else: return 2
        for ngram in self.ngrams(target_word):
            words = self.ngram_words[ngram]
            for word in words:
                if len(word) >= len(target_word) - self.len_variance and \
                   len(word) <= len(target_word) + self.len_variance:
                    word_ranking[word] += 1
        ranked_word_pairs = sorted(word_ranking.items(), key=itemgetter(1), reverse=True)
        if ranked_word_pairs != []:
            if self.lookup(target_word):
                    return target_word
            else:
                return ranked_word_pairs[0][0]
        else:
            return 1

def change(o):
    a=''
    for x in o:
        if x != 'lose':
            a=a+" "+x
    return ''.join(a)[1:]
def conn(arg1,arg2,arg3):
    if arg1 == None and arg2==1:
        return ''
    elif arg1 == None and arg2!=1:
        return arg3
    else:return change(arg1) 
    
def shopcorrection(o):
    sym_spell1 = SymSpell(max_dictionary_edit_distance=5)
    dictionary_path1 = 'shopcorr.txt'
    #dictionary_path = 'word_combine_lower.txt'
    #dictionary_path = 'word_supervised.txt'
    #dictionary_path = 'word_withshop.txt'
    sym_spell1.load_dictionary(dictionary_path1, 0, 1, separator="$")
    w=[]
    for a in o:
        suggestions = sym_spell1.lookup(a, Verbosity.CLOSEST, max_edit_distance=5)
        if suggestions:
            w.append(suggestions[0].term)
        #else: w.append(a)
    return w
def appen(arg1,arg2,arg3):
    if arg1 !='':
        
        s=''.join(change(arg1))+arg2
    else: s=arg2
    if arg3 !=[]:
        s=s+arg3
    return s

spellchecker=Autocorrect()
df['suggest']=df['final'].apply(spellchecker.suggested_words)
df['result']=df.apply(lambda row: conn(row['same_result'],row['final'],row['suggest']), axis = 1)
df['corr_shop']=df.apply(lambda row: shopcorrection(row['shop']), axis = 1)
df['corr_shop']=df['corr_shop'].apply(change)
df['percentage']=df['percentage'].apply(lambda x:''.join(x.split(' ')))
df['result1']=df['result'].apply(lambda x: [word.capitalize() for word in re.split(' ',x)if word not in stop_word])
df['corrected']=df['corr_shop'].map(str)+" "+df['result1'].apply(change)+" "+df['percentage'].map(str)
#df['corrected']=df['corrected'].apply(lambda x: x.replace('[','').replace(']','').replace('',''))
df