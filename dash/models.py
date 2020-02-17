from django.db import models
from django.contrib.auth.models import User
from PIL import Image


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username}Profile'

    def save(self, *args, **kwargs):
        super(Profile,self).save(*args, **kwargs)

        img = Image.open(self.image.path)
        if img.height > 600 or img.width > 600:
            output_size = (600, 600)
            img.thumbnail(output_size)
            img.save(self.image.path)


class Student(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    Postal_Code = models.CharField(max_length=100)
    City = models.CharField(max_length=100)
    Country = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.id} {self.Postal_Code} {self.City} {self.Country}'


class College(models.Model):
    id = models.AutoField(primary_key=True)
    Student_id = models.CharField(max_length=100)
    Stream = models.CharField(max_length=100)
    Course_year = models.CharField(max_length=100)
    Campus = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.id} {self.Student_id} {self.Stream} {self.Course_year} {self.Campus} '


class Internship(models.Model):
    id = models.AutoField(primary_key=True)
    Course = models.CharField(max_length=100)
    Year = models.CharField(max_length=100)
    Company_Name = models.CharField(max_length=100)
    Postal_Code = models.CharField(max_length=100)
    City = models.CharField(max_length=100)
    Country = models.CharField(max_length=100)
    Subject = models.CharField(max_length=5000)
    Pay_Details = models.FloatField()
    Student_id = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.Course} {self.Year} {self.City} {self.Company_Name} {self.Postal_Code} {self.City} {self.Country} {self.Subject} {self.Pay_Details} {self.Student_id}'


class Country(models.Model):
    id = models.AutoField(primary_key=True)
    country = models.CharField(max_length=100)
    latitude = models.FloatField(max_length=100)
    longitude = models.FloatField(max_length=100)
    country_name = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.id} {self.country} {self.latitude} {self.longitude} {self.country_name} '

