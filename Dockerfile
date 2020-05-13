FROM ubuntu:18.04

RUN adduser --disabled-password --gecos '' api-user

RUN apt-get update && yes|apt-get upgrade \
&& apt-get install -y wget bzip2 \
&& wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh \
&& bash Anaconda3-5.0.1-Linux-x86_64.sh -b \
&& rm Anaconda3-5.0.1-Linux-x86_64.sh# Set path to conda

ENV PATH /root/anaconda3/bin:$PATH

RUN conda update conda \
&& conda update anaconda \
&& conda install -c numba numba \
&& conda install -c conda-forge ffmpeg \
&& conda install -c conda-forge librosa \
&& conda update --all

# Add and install Python modules
ADD requirements.txt /src/requirements.txt

RUN pip install --upgrade pip \
&& cd /src; pip install -r requirements.txt

RUN chmod +x run.sh \
&& chown -R api-user:api-user ./	

USER api-user

# Bundle app source
ADD . /src

# Expose
EXPOSE  5000

# Run
CMD ["bash", "/src/run.sh"]
