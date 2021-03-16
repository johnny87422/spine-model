FROM ubuntu:latest

RUN mkdir /nginx

WORKDIR /nginx

RUN apt-get update && apt-get -y install sudo
RUN sudo apt-get update
RUN sudo apt-get install nginx -y
RUN sudo rm /etc/nginx/sites-available/default
RUN sudo rm /etc/nginx/sites-enabled/default
RUN sudo apt-get install vim -y

COPY nginx.conf /nginx/

RUN mv nginx.conf  /etc/nginx/conf.d/
#RUN mv nginx.conf  /etc/nginx/sites-enabled/

#RUN  sudo nginx -t
#RUN sudo nginx
#RUN bash -c "nginx -s reload"

EXPOSE 80

#CMD ["nginx", "-g", "daemon off;"]

