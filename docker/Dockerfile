FROM nvcr.io/nvidia/pytorch:21.02-py3

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

VOLUME [ "/data" ]
VOLUME [ "/code" ]

EXPOSE 8080

CMD ["jupyter", "notebook", "--port=8080", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
