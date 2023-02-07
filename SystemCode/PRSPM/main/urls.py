from django.urls import path
from . import views

urlpatterns = [
    # path("",views.index,name="index"),
    path('', views.index),
    path('create', views.index),
    path('next', views.index),
    path('final', views.index),
    path('patientData', views.index),
    path('User', views.User.as_view()),
    path('Text', views.Text.as_view()),
    path('CreateUsers', views.CreateUsers.as_view()),
    path('HospitalisedPrediction', views.HospitalisedPrediction.as_view()),
    path('CheckPatientData', views.CheckPatientData.as_view()),
    path('UpdateThreshold', views.UpdateThreshold.as_view()),
    path('CheckPatientID', views.CheckPatientID.as_view()),
    path('CheckPatientIDList', views.CheckPatientIDList.as_view()),
    path('CheckModelThreshold', views.CheckModelThreshold.as_view()),
    path('SaveVaccination', views.SaveVaccinationData.as_view()),
    path('ArticlePrediction', views.ArticlePrediction.as_view()),
]