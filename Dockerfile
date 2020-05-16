FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

# install package dependencies that can't be pip installed
RUN conda update conda \
&& conda install python=3.6 -y \
&& conda install -c conda-forge -y \
    ffmpeg \
    librosa \
&& apt-get install libsndfile1 -y \
&& conda clean --all

ENV PATH /opt/conda/envs/env/bin:$PATH

RUN adduser --disabled-password --gecos '' api-user \
&& pip install --no-cache-dir -r /app/requirements.txt \
&& chmod +x run.sh \
&& chown -R api-user:api-user ./	

USER api-user

EXPOSE  5000

# Run
CMD ["bash", "/app/run.sh"]
