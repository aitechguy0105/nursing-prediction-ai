FROM python:3.8.16-bullseye as base

ARG PROTOBUF_VERSION
ARG MODEL_TARGET_PATH="."

RUN apt update && apt install -y curl gnupg2 gcc \
    python3-dev build-essential unixodbc-dev libpq-dev \
    git wget libfontconfig1 libxrender1

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt update && ACCEPT_EULA=Y apt install -y msodbcsql17 mssql-tools

RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc && \
    echo 'echo PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile && \
    /bin/bash -c "source ~/.bashrc"

RUN apt install -y locales && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Install awscli so that we have the aws executable accessible
RUN pip install awscli==1.27.78


# Install dependencies
FROM python:3.8.16-bullseye as builder

ARG PROTOBUF_VERSION=4.23.3
ARG MODEL_TARGET_PATH="."
ARG AUTOMATIC_TRAINING_TARGET_PATH="src/training_pipeline"

RUN apt install -y git openssh-client && rm -rf /tmp/* /var/cache/apk/* &&\
    mkdir -p /root/.ssh && chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

ADD ./id_rsa ./id_rsa
RUN mv ./id_rsa /root/.ssh/id_rsa && chmod 0600 /root/.ssh/id_rsa

RUN pip install --cache-dir ./tmp/pipcache pipenv

RUN mkdir /deps
ADD ./id_rsa ./id_rsa
RUN mv ./id_rsa /root/.ssh/id_rsa && chmod 0600 /root/.ssh/id_rsa

COPY ${AUTOMATIC_TRAINING_TARGET_PATH}/Pipfile ${AUTOMATIC_TRAINING_TARGET_PATH}/Pipfile.lock /
RUN pipenv requirements > requirements-pipeline.txt
RUN pip install -r requirements-pipeline.txt --target /deps

COPY ${MODEL_TARGET_PATH}/Pipfile ${MODEL_TARGET_PATH}/Pipfile.lock /
RUN pipenv requirements > requirements-model.txt
RUN pip install -r requirements-model.txt --target /deps

# Install protofobuf separately to bypass locks
RUN pip install protobuf==${PROTOBUF_VERSION} --target /deps

RUN rm -f ./id_rsa


FROM base as runtime

ARG AUTOMATIC_TRAINING_TARGET_PATH="src/training_pipeline"

COPY --from=builder /deps /usr/local

ENV PYTHONPATH="/usr/local:/src:${PYTHONPATH}"

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY ${AUTOMATIC_TRAINING_TARGET_PATH}/docker-entrypoint.sh /usr/local/bin/

ADD /src /src

ENTRYPOINT ["docker-entrypoint.sh"]
