FROM tensorflow/tensorflow:1.12.3-gpu-py3

RUN pip install keras==2.2.4

RUN pip install matplotlib

COPY DCGAN_test.py /home

CMD python -u /home/DCGAN.py



