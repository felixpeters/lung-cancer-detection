FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

VOLUME [ "/data" ]
VOLUME [ "/code" ]

EXPOSE 8888