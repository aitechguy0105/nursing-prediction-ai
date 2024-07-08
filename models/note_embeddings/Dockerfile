# Base image
FROM continuumio/miniconda3:4.5.12

# apt-get and system utilities
RUN apt-get update && apt-get install --no-install-recommends -y curl apt-utils apt-transport-https gnupg2 gcc python3-dev build-essential unixodbc-dev

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 && ACCEPT_EULA=Y apt-get install -y mssql-tools
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile && \
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc && \
/bin/bash -c "source ~/.bashrc"

# install necessary locales
RUN apt-get install -y locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

# load in the environment.yml file - this file controls what Python packages we install
ADD environment.yml /

# install the Python packages we specified into the base environment
RUN conda update -n base conda -y && conda env update

# download the coder binary, untar it, and allow it to be executed
RUN wget https://github.com/cdr/code-server/releases/download/1.1156-vsc1.33.1/code-server1.1156-vsc1.33.1-linux-x64.tar.gz \
    && tar -xzvf code-server1.1156-vsc1.33.1-linux-x64.tar.gz && chmod +x code-server1.1156-vsc1.33.1-linux-x64/code-server

# install the Python extension
# RUN code-server1.1156-vsc1.33.1-linux-x64/code-server --user-data-dir /vscode-user-data --install-extension ms-python.python

COPY docker-entrypoint.sh /usr/local/bin/

ADD ./code /code
# ADD ./data /data

ENTRYPOINT ["docker-entrypoint.sh"]
