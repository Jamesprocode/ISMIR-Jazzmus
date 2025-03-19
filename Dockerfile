FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG GIT_AUTH_TOKEN
ARG GIT_USU
ARG GROUP_ID
ARG HUGGINGFACE_API_KEY
ARG UID
ARG USERNAME
ARG WANDB_API_KEY
ARG WORKDIR

RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt install imagemagick -y
RUN apt install git -y
RUN apt clean

RUN pip install --upgrade pip
RUN pip install pybind11 

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN rm requirements.txt

RUN groupadd -g $GROUP_ID $USERNAME && \
    useradd -m -u $UID -g $GROUP_ID -s /bin/bash $USERNAME

WORKDIR $WORKDIR


# RUN git clone https://github.com/JuanCarlosMartinezSevilla/ISMIR-Jazzmus.git .
RUN chown -R $USERNAME:$USERNAME $WORKDIR


USER $USERNAME
RUN wandb login $WANDB_API_KEY
# RUN echo "$HUGGINGFACE_API_KEY\n" | huggingface-cli login

COPY entrypoint.sh .
ENTRYPOINT ["bash", "entrypoint.sh"]