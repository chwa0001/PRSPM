# Generated by Django 3.1.7 on 2021-10-30 16:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='symptomspredicted',
            name='symptom',
            field=models.CharField(max_length=30000),
        ),
    ]
