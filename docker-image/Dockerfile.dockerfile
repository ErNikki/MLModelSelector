FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

#ARG CONDA_PYTHON_VERSION=3
#ARG CONDA_DIR=/opt/conda
ARG USERNAME=nik
ARG USERID=1000

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
#ENV PATH $CONDA_DIR/bin:$PATH
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda-installer.sh && \
#    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
#    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
#    rm -rf /tmp/*

# Create the user
#RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
#    chown $USERNAME $CONDA_DIR -R && \
#    adduser $USERNAME sudo && \
#    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

#USER $USERNAME
#WORKDIR /home/$USERNAME
WORKDIR /home

# Install mamba
#RUN conda install -y mamba -c conda-forge

#ADD ./torch-env.yml .
#RUN mamba env update --file ./torch-env.yml &&\
#    conda clean -tipy
RUN apt update
RUN apt-get -y install nvtop
RUN pip install matplotlib
RUN pip install torch-summary
RUN pip install gym
RUN apt-get -y install vim
RUN pip install pandas
RUN pip install scikit-learn
#RUN pip install keras
RUN pip install tensorflow==2.14
RUN pip install image-classifiers==1.0.0b1
#RUN pip install keras_applications

#COPY ./tesi /home/$USERNAME/tesi
#COPY ./tesi /home/tesi

# For interactive shell
#RUN conda init bash
#RUN echo "conda activate torch" >> /home/$USERNAME/.bashrc
