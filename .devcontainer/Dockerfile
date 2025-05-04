FROM mcr.microsoft.com/devcontainers/python:3.12

RUN python -m pip install --upgrade pip

COPY requirements.txt /tmp/

RUN pip3 install -r /tmp/requirements.txt
