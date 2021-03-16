from django.views.decorators.csrf import csrf_exempt
from django.conf.urls import include, url
from prediction import views

urlpatterns = [
    url(r'^prediction/', views.predict.as_view(), name='predict'),
    url(r'^change/', views.change.as_view(), name='change'),
]



