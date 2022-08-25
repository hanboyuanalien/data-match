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