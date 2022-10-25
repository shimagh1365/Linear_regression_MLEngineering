# python runtime
FROM python:3.9

# working directory
WORKDIR /

# install requirements
RUN pip3 install -r requirements.txt
 
# make port 8000 available to the world outside
EXPOSE 8000

COPY endpoint.py
COPY regressor.py

CMD export FLASK_APP=endpoint.py && flask run --host 0.0.0.0