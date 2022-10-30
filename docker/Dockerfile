FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Stockholm
RUN apt-get update && apt-get install -y python3-pip wget git default-jre

RUN useradd -d /home/quind -m quind
USER quind
WORKDIR /home/quind
RUN python3 --version
RUN pip install quinductor --user
RUN pip install dill --user
RUN pip install git+https://github.com/Maluuba/nlg-eval.git@master
RUN python3 -c "import stanza; stanza.download('sv')"
RUN .local/bin/nlg-eval --setup
