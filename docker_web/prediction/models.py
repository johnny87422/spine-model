from django.db import models
from mongoengine import *
# Create your models here.

#connect('test',host = '127.0.0.1',port = 27017)


class coordinate(Document):
    ID = StringField(max_length=30)
    x = ListField(FloatField())
    y = ListField(FloatField())

    meta = {'collection':'coordinate'}
    def __unicode__(self):
        return self.name
