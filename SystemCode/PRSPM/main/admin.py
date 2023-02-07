from django.contrib import admin
from .models import Patient, TextSymptoms, VaccinationData, SymptomsPredicted

# Register your models here.
admin.site.register(Patient)
admin.site.register(TextSymptoms)
admin.site.register(VaccinationData)
admin.site.register(SymptomsPredicted)
