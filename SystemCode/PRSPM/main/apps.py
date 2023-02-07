from django.apps import AppConfig
from .BertPrediction import load_model as BertModelLoad
from .TFIDFPrediction import load_model as TFIDFModelLoad
from .WordVecPrediction import load_model as WordVecModelLoad
from .HospitalisedPrediction import CovidHospitalizationPrediction
from .Covid19LitMining import classCovid19LitMining
import pickle


class MainConfig(AppConfig):
    name = 'main'

    with open("./main/model/Bert/classes_41.txt", "rb") as fp: 
        classes_41 = pickle.load(fp)
    BertModel = BertModelLoad('./main/model/Bert/bert_symptom_weights.h5')
    tfidfVectorizer,multiNomialModel,SVMModel = TFIDFModelLoad()
    WordVecDLModel,WordVecTokenizer = WordVecModelLoad()
    LitMiningModel = classCovid19LitMining('LitMiningModel')
    hospitalisedModel = CovidHospitalizationPrediction(classes_41)



    
    
