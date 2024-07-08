FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update

# Reference: https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15#ubuntu17
RUN apt-get install curl apt-transport-https gnupg2 -y
RUN apt-get install apt-transport-https
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install msodbcsql17 -y

RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile && \
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc && \
/bin/bash -c "source ~/.bashrc"

# install necessary locales
RUN apt-get install -y locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

RUN apt-get update && apt-get install -f python3-pyodbc build-essential unixodbc-dev libpq-dev python3-pip git -y
# Copying requirements. If any changes happen inside requirements file all the cache layers below this line will be
# invalidated
ADD requirements.txt /
RUN pip3 install -r requirements.txt
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install wget
# download the coder binary, untar it, and allow it to be executed
RUN wget https://github.com/cdr/code-server/releases/download/1.1156-vsc1.33.1/code-server1.1156-vsc1.33.1-linux-x64.tar.gz \
 && tar -xzvf code-server1.1156-vsc1.33.1-linux-x64.tar.gz && chmod +x code-server1.1156-vsc1.33.1-linux-x64/code-server


COPY docker-entrypoint.sh /usr/local/bin/

ADD ./src /src

ENTRYPOINT ["docker-entrypoint.sh"]