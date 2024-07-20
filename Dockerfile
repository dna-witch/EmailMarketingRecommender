FROM python:3.10-slim
COPY . ./
RUN pip3 install -r requirements.txt
CMD ["service.py"]
ENTRYPOINT ["python"]