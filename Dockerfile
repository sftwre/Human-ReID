FROM python:3.12.4-slim

# Set the working directory in the container
WORKDIR /usr/src/app

COPY requirements.txt ./
COPY *.py ./
COPY models ./models
COPY index ./index
COPY dashboards.sh ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 6006

EXPOSE 8080

RUN python data_prep.py

CMD [ "/bin/bash" ]