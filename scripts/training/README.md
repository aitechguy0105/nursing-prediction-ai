# Saiva Tools

### One time install on mac for sqlserver driver
- brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
- brew update
- HOMEBREW_NO_ENV_FILTERING=1 ACCEPT_EULA=Y brew install msodbcsql17 mssql-tools

### To setup the environment to run scripts/training
## One time setup
- `curl https://bootstrap.pypa.io/get-pip.py | python3`
- `pip install --user pipenv`
- `cd saivahc/scripts/training/`
- `pipenv install`

Subsequent invocations
- `pipenv shell`

To deactivate environment
- `exit`

### Example script usage
- `cd scripts/training`
- `pipenv shell`
- `python copy_model_metadata.py`