# Generated by Django 3.1.7 on 2021-11-12 16:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0006_auto_20211111_0232'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='vaccinationdata',
            name='numHosDays',
        ),
    ]