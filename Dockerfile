FROM python:3.9-slim-buster
COPY src/automatise/requirements.txt /tmp/
COPY ./src /app
WORKDIR "/app"
RUN pip3 install -r /tmp/requirements.txt
ENTRYPOINT [ "python3" ]
CMD [ "src/automatise/app.py" ]
EXPOSE 8050