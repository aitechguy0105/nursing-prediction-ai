# Saiva Tools

### One time install on mac for sqlserver driver
- brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
- brew update
- HOMEBREW_NO_ENV_FILTERING=1 ACCEPT_EULA=Y brew install msodbcsql17 mssql-tools

## One time setup for pipenv
- `curl https://bootstrap.pypa.io/get-pip.py | python3`
- `pip install --user pipenv`

### To setup the environment to run tools
One time First time setup (if you don't have a virtualenv yet):
- cd tools
- pipenv install

Subsequent invocations
- pipenv shell

To deactivate environment
- deactivate

### Example script usage
- cd tools
- pipenv shell
- python db_backup_restore.py