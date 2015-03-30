# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
from nltk.corpus import sentiwordnet as swn
doc="Nice and friendly place with excellent food and friendly and helpful staff. You need a car though. The children wants to go back! Playground and animals entertained them and they felt like at home. I also recommend the dinner! Great value for the price!"
sentences = nltk.sent_tokenize(doc)
stokens = [nltk.word_tokenize(sent) for sent in sentences]
taggedlist=[]
for stoken in stokens:        
     taggedlist.append(nltk.pos_tag(stoken))
wnl = nltk.WordNetLemmatizer()

score_list=[]
for idx,taggedsent in enumerate(taggedlist):
    score_list.append([])
    for idx2,t in enumerate(taggedsent):
        newtag=''
        lemmatized=wnl.lemmatize(t[0])
        if t[1].startswith('NN'):
            newtag='n'
        elif t[1].startswith('JJ'):
            newtag='a'
        elif t[1].startswith('V'):
            newtag='v'
        elif t[1].startswith('R'):
            newtag='r'
        else:
            newtag=''       
        if(newtag!=''):    
            synsets = list(swn.senti_synsets(lemmatized, newtag))
            #Getting average of all possible sentiments, as you requested        
            score=0
            if(len(synsets)>0):
                for syn in synsets:
                    score+=syn.pos_score()-syn.neg_score()
                score_list[idx].append(score/len(synsets))
            
print(score_list)
sentence_sentiment=[]

for score_sent in score_list:
    sentence_sentiment.append(sum([word_score for word_score in score_sent])/len(score_sent))
print("Sentiment for each sentence for:"+doc)
print(sentence_sentiment)




