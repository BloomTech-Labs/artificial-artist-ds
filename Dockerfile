from python:3.6

Run pip3 install --upgrade pip \
&& apt-get update \
&& apt-get install ffmpeg -y \
&& apt-get install libsndfile1 -y

WORKDIR /artificial-artist-ds

COPY . /artificial-artist-ds

RUN pip3 --no-cache-dir install --ignore-installed -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["application.py"] 