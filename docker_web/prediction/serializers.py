from rest_framework_mongoengine import serializers
from .models import coordinate

class coordinateSerializer(serializers.DocumentSerializer): 
    class Meta:
        model = coordinate
        fields = '__all__'
