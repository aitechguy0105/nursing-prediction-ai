#!/bin/bash
set -e

if [ $# -eq 0 ]
  then
    pipenv run jupyter lab --ip=0.0.0.0  --NotebookApp.token='local-development' --allow-root --no-browser &> /dev/null &
    code-server1.1156-vsc1.33.1-linux-x64/code-server --allow-http --no-auth --user-data-dir /vscode-user-data /src
  else
    exec "$@"
fi 
