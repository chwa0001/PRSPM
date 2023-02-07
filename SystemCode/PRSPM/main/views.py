from django.shortcuts import render

from rest_framework import generics,status
from rest_framework.views import APIView
from rest_framework.response import Response

import pickle
from ast import literal_eval
from keras.preprocessing.sequence import pad_sequences

from .serializers import PatientSerializer, TextSymptomsSerializer,VaccinationDataSerializer, SymptomsPredictedSerializer,ThresholdSerializer

from .models import Patient,TextSymptoms,SymptomsPredicted,VaccinationData,Threshold
from .TextPreprocessing import text_preprocessing
from .BertPrediction import make_prediction as BertPrediction
from .BertPrediction import prepare_bert_input as BertInput
from .apps import MainConfig
from .Covid19LitMining import fcnCovid19LitMining
import numpy as np


# Create your views here.
def index(request, *args, **kwargs):
    return render(request, 'ui/index.html')

def TextMiningPrediction(text,preprocessedtext,bertThreshold=0.3,WordVecThreshold=0.3,TFDIDFThreshold=0.3):
    #tfidf
    sentences = np.asarray([preprocessedtext])
    X = MainConfig.tfidfVectorizer.transform(sentences)
    svm_predicts =  MainConfig.SVMModel.predict(X)
    symptomListTfidf = []
    thresh = 0.3
    for sentence, pred in zip(sentences, svm_predicts):
        pred = pred.tolist()
        for index,score in enumerate(pred):
            if score>thresh:
                symptom = MainConfig.classes_41[index]
                symptomListTfidf.append(symptom)
    
    #WordVec
    df_symptom_text_list = [preprocessedtext]
    full_symptoms_tokenized = MainConfig.WordVecTokenizer.texts_to_sequences(df_symptom_text_list)
    input = pad_sequences(full_symptoms_tokenized, maxlen=2845, padding='post')
    WordVecPredicts = MainConfig.WordVecDLModel.predict(input)
    symptomListWordVec = []
    thresh = WordVecThreshold
    for sentence, pred in zip(input, WordVecPredicts):
        pred = pred.tolist()
        for score in pred:
            if score>thresh:
                i = pred.index(score)
                symptom = MainConfig.classes_41[i]
                symptomListWordVec.append(symptom)
    #Bert
    sentences = np.asarray([text])
    token = BertInput(sentences)
    prediction = MainConfig.BertModel.predict(token)
    symptomListBert = []
    topn = 4
    thresh = bertThreshold
    for sentence, pred in zip(sentences, prediction):
        pred = pred.tolist()
        for score in pred:
            if score>thresh:
                i = pred.index(score)
                symptom = MainConfig.classes_41[i]
                symptomListBert.append(symptom)
    symptomlist = sorted(np.unique(symptomListBert+symptomListWordVec+symptomListTfidf))
    return {'Bert':symptomListBert,'WordVec':symptomListWordVec,'Tfidf':symptomListTfidf,'final':symptomlist}



class User(generics.CreateAPIView):
    queryset            =  Patient.objects.all()
    serializer_class    =  PatientSerializer

class Text(generics.CreateAPIView):
    queryset            =  TextSymptoms.objects.all()
    serializer_class    =  TextSymptomsSerializer

class CreateUsers(APIView):
    patient_serializer_class = PatientSerializer
    def post(self, request, format=None):
        try:
            print(request.data.keys())
            if 'patientID' and 'name' and 'age' and 'gender' and 'text' in request.data.keys():
                queryPatient = Patient.objects.filter(patientID=request.data['patientID'])
                if queryPatient.exists():
                    patientData = queryPatient[0]
                    Response({'Bad Request': 'Patient '+ patientData.name + ' have medical record in the system.'}, status=status.HTTP_400_BAD_REQUEST)
                elif len(request.data['name'])>0 and len(str(request.data['age']))>0 and len(request.data['gender'])>0:
                    newPatient = Patient(patientID=request.data['patientID'],name=request.data['name'],age=request.data['age'],gender = request.data['gender'] )
                    newPatient.save()
                    preprocessedtext = text_preprocessing(request.data['text'])
                    newTextSymptoms = TextSymptoms(patientKey=newPatient,text=request.data['text'],processedText=preprocessedtext)
                    newTextSymptoms.save()
                    bertThreshold = Threshold.objects.get(name='Bert').thresholdValue
                    WordVecThreshold = Threshold.objects.get(name='WordVec').thresholdValue
                    TFDIDFThreshold = Threshold.objects.get(name='TFIDF').thresholdValue
                    dictSymptom = TextMiningPrediction(request.data['text'],preprocessedtext,bertThreshold=bertThreshold,WordVecThreshold=WordVecThreshold,TFDIDFThreshold=TFDIDFThreshold)
                    newPredictedSymptoms = SymptomsPredicted(patientKey=newPatient,symptom=str(dictSymptom['final']),symptomBert=str(dictSymptom['Bert']),symptomTFIDF=str(dictSymptom['WordVec']),symptomWordVec=str(dictSymptom['Tfidf']))
                    newPredictedSymptoms.save()
                    return Response(PatientSerializer(newPatient).data, status=status.HTTP_201_CREATED)

            return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'Bad Request': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class HospitalisedPrediction(APIView):
    symptoms_serializer_class = TextSymptomsSerializer
    def get(self, request, format=None):
        try:
            patientID = request.GET.get('id')
            if patientID.isnumeric():
                queryPatient = Patient.objects.filter(patientID=patientID)
                if queryPatient.exists():
                    patientData = queryPatient[0]
                    datasetToFit = {}
                    datasetToFit['AGE_YRS'] = [int(patientData.age)]
                    datasetToFit['SEX'] = [str(patientData.gender)]
                    vaccinationData = VaccinationData.objects.filter(patientKey=patientData.id)[0]
                    datasetToFit['VAX_MANU'] = [str(vaccinationData.vaccineBrand)]
                    datasetToFit['VAX_DOSE_SERIES'] = ['UNK' if vaccinationData.numDoses<1 else ('3+' if vaccinationData.numDoses>3 else str(vaccinationData.numDoses))]
                    datasetToFit['NUMDAYS'] = int(vaccinationData.numDays)
                    datasetToFit['BIRTH_DEFECT'] = int(vaccinationData.birthDefect)
                    datasetToFit['DISABLE'] = int(vaccinationData.disability)
                    datasetToFit['OFC_VISIT'] = int(vaccinationData.visit)
                    datasetToFit['L_THREAT'] = int(vaccinationData.lifeDesease)
                    symptomsPredicted = SymptomsPredicted.objects.filter(patientKey=patientData.id)[0]
                    datasetToFit['symptoms'] = symptomsPredicted.symptom
                    predicts = MainConfig.hospitalisedModel.makePrediction(datasetToFit)
                    thresh = Threshold.objects.get(name='Hospitalised').thresholdValue
                    return Response({'predictionRate':predicts[0][0],'threshold':thresh}, status=status.HTTP_201_CREATED)
                else:
                    return Response({'Bad Request': 'Patient ID has been occupied by User: '+queryPatient[0].name}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({'Bad Request': 'Patient ID is invalid.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'Bad Request': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class SaveVaccinationData(APIView):
    vaccine_serializer_class = VaccinationDataSerializer
    def post(self, request, format=None):
        try:
            if 'patientID' and 'vaccineBrand' and 'numDoses' and 'numDays' and 'birthDefect' and 'disability' and 'visit' and 'desease' in request.data.keys():
                queryPatient = Patient.objects.filter(patientID=request.data['patientID'])
                if queryPatient.exists():
                    patientData = queryPatient[0]
                    queryVaccinationData = VaccinationData.objects.filter(patientKey=patientData.id)
                    if queryVaccinationData.exists():
                        VaccineData = queryVaccinationData[0]
                        VaccineData.vaccineBrand = request.data['vaccineBrand']
                        VaccineData.numDoses = request.data['numDoses']
                        VaccineData.numDays = request.data['numDays']
                        VaccineData.birthDefect = request.data['birthDefect']
                        VaccineData.disability = request.data['disability']
                        VaccineData.visit = request.data['visit']
                        VaccineData.lifeDesease = request.data['desease']
                        VaccineData.save(update_fields=['vaccineBrand','numDoses','numDays','birthDefect','disability','visit','lifeDesease'])
                        return Response(VaccinationDataSerializer(VaccineData).data, status=status.HTTP_201_CREATED)
                    else:
                        patientKey = Patient.objects.get(id=patientData.id)
                        newVaccineData = VaccinationData(patientKey=patientKey,vaccineBrand=request.data['vaccineBrand'],numDoses=request.data['numDoses'],numDays=request.data['numDays'],birthDefect=request.data['birthDefect'],disability=request.data['disability'],visit=request.data['visit'],lifeDesease=request.data['desease'])
                        newVaccineData.save()
                        return Response(VaccinationDataSerializer(newVaccineData).data, status=status.HTTP_201_CREATED)
                else:
                    return Response({'Bad Request': 'Patient ID is invalid.'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'Bad Request': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class CheckPatientData(APIView):
    symptoms_serializer_class = TextSymptomsSerializer
    def get(self, request, format=None):
        try:
            patientID = request.GET.get('id')
            if patientID.isnumeric():
                queryPatient = Patient.objects.filter(patientID=patientID)
                if queryPatient.exists():
                    patientData = queryPatient[0]
                    dictPatient = {'name':patientData.name,'age':patientData.age,'gender':patientData.gender}
                    vaccinationData = VaccinationData.objects.filter(patientKey=patientData.id)[0]
                    dictVaccinationData = {'vaccineBrand':vaccinationData.vaccineBrand,'numDoses':vaccinationData.numDoses,'numDays':vaccinationData.numDays,'birthDefect':vaccinationData.birthDefect,'disability':vaccinationData.disability,'visit':vaccinationData.visit,'lifeDesease':vaccinationData.lifeDesease}
                    symptomsPredicted = SymptomsPredicted.objects.filter(patientKey=patientData.id)[0]
                    dictSymptomsPredicted = {'symptom':[symptom.strip('SYMPTOMS') for symptom in literal_eval(symptomsPredicted.symptom)],'symptomBert':[symptom.strip('SYMPTOMS') for symptom in literal_eval(symptomsPredicted.symptomBert)],'symptomTFIDF':[symptom.strip('SYMPTOMS') for symptom in literal_eval(symptomsPredicted.symptomTFIDF)],'symptomWordVec':[symptom.strip('SYMPTOMS') for symptom in literal_eval(symptomsPredicted.symptomWordVec)]}
                    textSymptoms = TextSymptoms.objects.filter(patientKey=patientData.id)[0]
                    dictTextSymptoms = {'text':textSymptoms.text}
                    
                    return Response({'patientData':{'patient':dictPatient,'textSymptoms':dictTextSymptoms,'vaccinationData':dictVaccinationData,'symptomsPredicted':dictSymptomsPredicted} }, status=status.HTTP_201_CREATED)
                else:
                    return Response({'Bad Request': 'Patient ID has been occupied by User: '+queryPatient[0].name}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({'Bad Request': 'Patient ID is invalid.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'Bad Request': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class CheckPatientID(APIView):
    symptoms_serializer_class = TextSymptomsSerializer
    def get(self, request, format=None):
        try:
            patientID = request.GET.get('id')
            if patientID.isnumeric():
                queryPatient = Patient.objects.filter(patientID=patientID)
                if queryPatient.exists():
                    return Response({'Bad Request': 'Patient ID has been occupied by User: '+queryPatient[0].name}, status=status.HTTP_400_BAD_REQUEST)
                else:
                    return Response({'HTTP STATUS OK': 'Patient ID is not used.'}, status=status.HTTP_201_CREATED)
            else:
                return Response({'Bad Request': 'Patient ID is invalid.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'Bad Request': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class CheckPatientIDList(APIView):
    def get(self, request, format=None):
        try:
            patientIDList = Patient.objects.values_list('patientID',flat=True)
            return Response({'patientList': patientIDList}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'Bad Request': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class CheckModelThreshold(APIView):
    symptoms_serializer_class = ThresholdSerializer
    def get(self, request, format=None):
        try:
            dictThreshold = {}
            for model in ['Bert','WordVec','TFIDF','Hospitalised']:
                modelData = Threshold.objects.get(name=model)
                dictThreshold[model] = modelData.thresholdValue
            return Response(dictThreshold, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response({'Bad Request': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class UpdateThreshold(APIView):
    threshold_serializer_class = ThresholdSerializer
    def post(self, request, format=None):
        try:
            if 'name' in request.data.keys():
                if request.data['name'] in ['Bert','WordVec','TFIDF','Hospitalised']:
                    queryModel = Threshold.objects.filter(name=request.data['name'])
                    if queryModel.exists():
                        modelThresholdData = queryModel[0]
                        modelThresholdData.name = request.data['name']
                        modelThresholdData.thresholdValue = request.data['thresholdValue']
                        modelThresholdData.save(update_fields=['name', 'thresholdValue'])
                        return Response(ThresholdSerializer(modelThresholdData).data, status=status.HTTP_201_CREATED)
            return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as error:
            return Response({"Bad Request": str(error)}, status=status.HTTP_400_BAD_REQUEST)



class ArticlePrediction(APIView):
    symptoms_serializer_class = SymptomsPredictedSerializer
    def get(self, request, format=None):
        patientID = request.GET.get('id')
        if patientID.isnumeric():
            queryPatient = Patient.objects.filter(patientID=patientID)
            if queryPatient.exists():
                patientData = queryPatient[0]
                symtomsPredictedData = SymptomsPredicted.objects.filter(patientKey=patientData.id)
                if symtomsPredictedData.exists():
                    predictedSymtoms = symtomsPredictedData[0]
                    symptomList = [symptoms.strip("SYMPTOMS") for symptoms in literal_eval(predictedSymtoms.symptom)]
                    feedback = MainConfig.LitMiningModel.Covid19LitMining(symptomList)
                    return Response({'ArticlePredicted':feedback}, status=status.HTTP_200_OK)
            else:
                return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
        
