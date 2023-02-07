from rest_framework  import serializers
from .models import Patient,TextSymptoms,VaccinationData,SymptomsPredicted,Threshold

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ('id','patientID','name','age','gender')

class TextSymptomsSerializer(serializers.ModelSerializer):
    class Meta:
        model = TextSymptoms
        fields = ('id','patientKey','text','processedText')

class VaccinationDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = VaccinationData
        fields = ('id','patientKey','vaccineBrand','numDoses','numDays','birthDefect','disability','visit','lifeDesease')

class SymptomsPredictedSerializer(serializers.ModelSerializer):
    class Meta:
        model = SymptomsPredicted
        fields = ('id','patientKey','symptom','symptomBert','symptomTFIDF','symptomWordVec')

class ThresholdSerializer(serializers.ModelSerializer):
    class Meta:
        model = Threshold
        fields = ('id','name','thresholdValue')