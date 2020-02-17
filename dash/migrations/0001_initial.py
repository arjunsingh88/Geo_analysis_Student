# Generated by Django 2.2.1 on 2019-05-31 12:57

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='College',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('Student_id', models.CharField(max_length=100)),
                ('Stream', models.CharField(max_length=100)),
                ('Course_year', models.CharField(max_length=100)),
                ('Campus', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Country',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('country', models.CharField(max_length=100)),
                ('latitude', models.FloatField(max_length=100)),
                ('longitude', models.FloatField(max_length=100)),
                ('country_name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Internship',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('Course', models.CharField(max_length=100)),
                ('Year', models.CharField(max_length=100)),
                ('Company_Name', models.CharField(max_length=100)),
                ('Postal_Code', models.CharField(max_length=100)),
                ('City', models.CharField(max_length=100)),
                ('Country', models.CharField(max_length=100)),
                ('Subject', models.CharField(max_length=5000)),
                ('Pay_Details', models.FloatField()),
                ('Student_id', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('Postal_Code', models.CharField(max_length=100)),
                ('City', models.CharField(max_length=100)),
                ('Country', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(default='default.jpg', upload_to='profile_pics')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
