import pandas as pd
import numpy as np
import matplotlib as plt
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from .TextPreprocessing import dummy_fun

torch.cuda.set_device(0)
device = torch.device('cpu')

#function method for model initialization
def fcnCovid19LitMining(symptom = 'fever'):
    # sbert embedder
    embedder = SentenceTransformer('msmarco-distilbert-base-v4').to(device)
    # QA model
    tokenizer = AutoTokenizer.from_pretrained("gerardozq/biobert_v1.1_pubmed-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("gerardozq/biobert_v1.1_pubmed-finetuned-squad").to(device)


    filename = './main/model/Covid19LitMining/tfidf_doc'
    infile = open(filename,'rb')
    tfidf_doc = pickle.load(infile)
    infile.close()

    doc_list = tfidf_doc['doc_list']
    doc_list_word = tfidf_doc['doc_list_word']

    def dummy_fun(doc):
        return doc
    tfidf = TfidfVectorizer(analyzer='word',tokenizer=dummy_fun,preprocessor=dummy_fun,token_pattern=None)
    tfidf_matrix = tfidf.fit_transform(doc_list)

    ## SBERT processing

    filename = './main/model/Covid19LitMining/sbert_doc'
    infile = open(filename,'rb')
    sbert_doc = pickle.load(infile)
    infile.close()

    paracorp = sbert_doc['paracorp']
    para_doc = sbert_doc['para_doc']

                    
    filename = './main/model/Covid19LitMining/emb_corpus300'
    infile = open(filename,'rb')
    corpus_embeddings1 = pickle.load(infile)
    infile.close()

    corpus = paracorp

    output = {}
    i = 1
    
    # Auto-query generation
    query = 'is ' + str(symptom) + ' caused by vaccine a severe adverse effect'
    query_token = tokenizer(query)
    query_vec = tfidf.transform([query_token['input_ids'][1:len(query_token['input_ids'])-1]])
    cosine_sim = cosine_similarity(tfidf_matrix, query_vec)
    tfidf_score = torch.FloatTensor(np.transpose(cosine_sim)[0],device="cpu")
    
    # sbert score
    top_k = min(5, len(corpus))

    query_embedding = embedder.encode(query, convert_to_tensor=True,device='cpu')

    # Linear combination of tfidf and sbert scores to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings1)[0]
    combined_scores = 0.5*cos_scores + 0.5*tfidf_score
    top_results = torch.topk(combined_scores, k=top_k)

    #for score, idx in zip(top_results[0], top_results[1]):
        #print(corpus[idx], "(Score: {:.4f})".format(score), "idx: " + str(idx))
    
    for idx in top_results[1]:
        # QA search
        question = query
        text = corpus[idx]
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        
        # Sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        sentence = answer 
        vs = analyzer.polarity_scores(sentence)
        
        output[i] = {}
        output[i]['text'] =  str(para_doc[idx.item()]['text'])
        output[i]['title'] =  str(para_doc[idx.item()]['title'])
        output[i]['authors'] =  str(para_doc[idx.item()]['authors'])
        output[i]['publish_time'] =  str(para_doc[idx.item()]['publish_time'])
        output[i]['journal'] =  str(para_doc[idx.item()]['journal'])
        output[i]['sentiment'] =  vs
        i = i + 1
        
    return output

#class method for model initialization
class classCovid19LitMining():

    def __init__(self,name):
        self.name = name

        #Load pickled Model Method
        # with open("./main/model/Covid19LitMining/models_litmining", "rb") as fp:
        #     self.models_litmining1 = pickle.load(fp)
        # self.tokenizer = self.models_litmining1['tokenizer']
        # self.model = self.models_litmining1['model']
        # self.embedder = self.models_litmining1['embedder']

        #Redefine transformer and tokenizer method
        self.embedder = SentenceTransformer('msmarco-distilbert-base-v4').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("gerardozq/biobert_v1.1_pubmed-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("gerardozq/biobert_v1.1_pubmed-finetuned-squad").to(device)

        
        with open("./main/model/Covid19LitMining/tfidf_doc", "rb") as fp:
            self.tfidf_doc = pickle.load(fp)
        self.doc_list = self.tfidf_doc['doc_list']
        self.doc_list_word = self.tfidf_doc['doc_list_word']

        

        self.tfidf = TfidfVectorizer(analyzer='word',tokenizer=dummy_fun,preprocessor=dummy_fun,token_pattern=None)
        self.tfidf_matrix = self.tfidf.fit_transform(self.doc_list)
        with open("./main/model/Covid19LitMining/sbert_doc", "rb") as fp:
            self.sbert_doc = pickle.load(fp)
        self.paracorp = self.sbert_doc['paracorp']
        self.para_doc = self.sbert_doc['para_doc']

        with open("./main/model/Covid19LitMining/emb_corpus300", "rb") as fp:
            self.corpus_embeddings1 = pickle.load(fp)
        self.corpus = self.paracorp
    
    def Covid19LitMining(self,symptomInput):
        symptomlist = []
        rejectlist = ['covid-19', 'blood test','computerised tomogram','sars-cov-2 test positive',
                    'cerebrovascular accident','electrocardiogram','echocardiogram','troponin increased',
                    'arthralgia','myalgia','hyperhidrosis','paraesthesia','hypoaesthesia','feeling abnormal']
        for symptom in symptomInput:
            if symptom == 'dyspnoea':
                symptom1 = 'dyspnea'
            elif symptom == 'pyrexia':
                symptom1 = 'fever'
            elif symptom == 'injection site erythema':
                symptom1 = 'erythema'
            elif symptom == 'myalgia':
                symptom1 = 'myalgias'
            elif symptom == 'lymphadenopathy':
                symptom1 = 'lymph nodes'
            elif symptom in rejectlist:
                symptom1 = 'pain'
            else:
                symptom1 = symptom
            
            if symptom1 not in symptomlist:
                symptomlist.append(symptom1)
            
        
        output = {}
        i = 1
        
        resultlist_value = torch.empty((0))
        resultlist_indices = torch.empty((0))
        query_symp = []

        for symp in symptomlist:
            # Auto-query generation
            query = 'is ' + str(symp) + ' caused by vaccine a severe adverse effect'
            #queries = ['is fever caused by pfizer vaccine a severe adverse effect']
            # tfidf score
            query_token = self.tokenizer(query)
            query_vec = self.tfidf.transform([query_token['input_ids'][1:len(query_token['input_ids'])-1]])
            cosine_sim = cosine_similarity(self.tfidf_matrix, query_vec)
            tfidf_score = torch.FloatTensor(np.transpose(cosine_sim)[0],device='cpu')

            # sbert score
            top_k = min(5, len(self.corpus))

            query_embedding = self.embedder.encode(query, convert_to_tensor=True,device='cpu')

            # Linear combination of tfidf and sbert scores to find the highest 5 scores
            cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings1)[0]
            combined_scores = 0.7*cos_scores + 0.3*tfidf_score
            top_results = torch.topk(combined_scores, k=top_k)
            resultlist_value = torch.cat((resultlist_value,top_results[0]))
            resultlist_indices = torch.cat((resultlist_indices,top_results[1]))
            query_symp.extend([symp for i in range(5)])
        
        top_results_list = torch.topk(resultlist_value, k=top_k)
        symp_idx = top_results_list[1].tolist()
        query_symp = [query_symp[i] for i in symp_idx]
        qi = 0
        #for score, idx in zip(top_results[0], top_results[1]):
            #print(corpus[idx], "(Score: {:.4f})".format(score), "idx: " + str(idx))
        
        for idx in resultlist_indices[top_results_list[1]].int():
            # QA search
            question = 'is ' + query_symp[qi] + ' caused by vaccine a severe adverse effect'
            text = self.corpus[idx]
            inputs = self.tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            outputs = self.model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            # Get the most likely beginning of answer with the argmax of the score
            answer_start = torch.argmax(answer_start_scores)
            # Get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            
            # Sentiment analyzer
            analyzer = SentimentIntensityAnalyzer()
            sentence = answer 
            vs = analyzer.polarity_scores(sentence)
            
            output[i] = {}
            output[i]['name'] = 'article '+str(i)
            output[i]['text'] =  self.para_doc[idx.item()]['text']
            output[i]['title'] =  self.para_doc[idx.item()]['title']
            output[i]['authors'] =  self.para_doc[idx.item()]['authors']
            output[i]['publish_time'] = self.para_doc[idx.item()]['publish_time'].strftime('%m/%d/%Y')     
            output[i]['journal'] =  self.para_doc[idx.item()]['journal']
            output[i]['sentiment'] =  vs
            i = i + 1
            qi = qi + 1
            
        return output