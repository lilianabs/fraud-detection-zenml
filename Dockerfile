FROM zenmldocker/zenml

WORKDIR /predict-hotel-reserv-cancelation

ADD . /predict-hotel-reserv-cancelation

RUN pip install -r requirements.txt

CMD ["python", "run_training_pipeline.py"]