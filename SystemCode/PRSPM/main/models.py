from django.db import models

# Create your models here.

class Patient(models.Model):
    patientID       = models.IntegerField(null=False, default=0,unique=True)
    name            = models.CharField(max_length=200)
    age             = models.IntegerField(null=False, default=0)
    gender          = models.CharField(max_length=1, default='')

    class Meta:
        managed = True
        db_table    = "USER_INFORMATION_DATABASE"


class TextSymptoms(models.Model):
    patientKey      = models.ForeignKey(Patient, on_delete=models.CASCADE)
    text            = models.CharField(max_length=30000)
    processedText   = models.CharField(max_length=30000)

    class Meta:
        managed = True
        db_table    = "USER_TEXTSYMTOMS_DATABASE"

class VaccinationData(models.Model):
    patientKey       = models.ForeignKey(Patient, on_delete=models.CASCADE)
    vaccineBrand     = models.CharField(max_length=200)
    numDoses         = models.IntegerField(null=False, default=0)
    numDays          = models.IntegerField(null=False, default=0)
    # numHosDays       = models.IntegerField(null=False, default=0)
    birthDefect      = models.BooleanField(null=False, default=False)
    disability       = models.BooleanField(null=False, default=False)
    visit            = models.BooleanField(null=False, default=False)
    lifeDesease      = models.BooleanField(null=False, default=False)

    class Meta:
        managed = True
        db_table    = "USER_VACCINATION_DATABASE"

class SymptomsPredicted(models.Model):
    patientKey      = models.ForeignKey(Patient, on_delete=models.CASCADE)
    symptom         = models.CharField(max_length=30000)
    symptomBert     = models.CharField(max_length=30000)
    symptomTFIDF    = models.CharField(max_length=30000)
    symptomWordVec  = models.CharField(max_length=30000)

    class Meta:
        managed = True
        db_table    = "USER_SYMTOMS_DATABASE"

class Threshold(models.Model):
    name            = models.CharField(max_length=200)
    thresholdValue  = models.FloatField(null=False, default=0)

    class Meta:
        managed = True
        db_table    = "MODEL_THRESHOLD_DATABASE"