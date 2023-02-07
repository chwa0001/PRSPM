# Generated by Django 3.1.7 on 2021-11-10 18:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_auto_20211031_1917'),
    ]

    operations = [
        migrations.CreateModel(
            name='Threshold',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('thresholdValue', models.FloatField(default=0)),
            ],
        ),
        migrations.AddField(
            model_name='symptomspredicted',
            name='symptomBert',
            field=models.CharField(default='', max_length=30000),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='symptomspredicted',
            name='symptomTFIDF',
            field=models.CharField(default='', max_length=30000),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='symptomspredicted',
            name='symptomWordVec',
            field=models.CharField(default='', max_length=30000),
            preserve_default=False,
        ),
    ]