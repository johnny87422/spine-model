FROM python:3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get update
RUN apt-get -y install sudo
RUN sudo apt-get update
RUN apt-get update &&\
    apt-get install -y binutils libproj-dev gdal-bin
RUN pip install psycopg2-binary 
RUN sudo apt-get install python3-psycopg2
RUN pip install -r requirements.txt
RUN pip install numpy
RUN pip install tensorflow==1.12.0
RUN pip install opencv-python
RUN pip install requests
RUN pip install opencv-contrib-python
RUN pip install mongoengine
RUN pip install Django-rest-framework-mongoengine
RUN pip install django-rest-framework-mongoengine==3.3.1
RUN pip install pymongo
RUN pip install django-cors-headers
RUN sudo apt-get install python3-matplotlib
RUN apt-get update
RUN pip install uwsgi
COPY . /code/
EXPOSE 3000