FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

# install package dependencies
RUN conda update conda \
&& conda install python=3.6 -y

ENV PATH /opt/conda/envs/env/bin:$PATH

RUN adduser --disabled-password --gecos '' api-user

RUN conda install -c conda-forge -y \
    ffmpeg \
    librosa \
&& apt-get install libsndfile1 -y

RUN pip install -r /app/requirements.txt

RUN chmod +x run.sh \
&& chown -R api-user:api-user ./	

USER api-user

# Expose
EXPOSE  5000

# Run
CMD ["bash", "/app/run.sh"]
