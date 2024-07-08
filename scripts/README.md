# Saiva Scripts



### Steps to be followed for staging release
- Add GIT tags for the build by executing `deployment/add_git_tag.sh`

- Trigger `deployment/deploy_staging.sh`. This script will trigger AWS CodeBuild Projects `ie. airflow_staging, etl_staging & prediction_models_staging` 

### Steps to be followed for prod release
- Trigger `deploy_prod.sh`. This will trigger AWS CodeBuild project `deploy_prod`


### Deployment Internals
* We have a Makefile for every docker container we build. This file helps us in pushing the docker images to AWS ECR
* We use AWS CodeBuild for tracking the deployment history 
* CodeBuild projects help us in making staging docker builds and restarting the ECS Services
* During prod release we use the staging docker Images and add `prod` tags to them. Also re-start the prod ECS services
* `scripts/prod_build/push_prod_builds.sh` script adds `prod` tags to our `staging` builds
* AWS CodeBuild project `deploy_prod` automates the process of updating the tags and re-starting the prod ECS services
* Name of new prediction Models needs to be added into `push_prod_builds.sh`

### Adding New client
* Add new client in /models/buildspec.yml
* Add new client into /scripts/prod_build/push_prod_builds.sh

### New version of Model
* when ever there is a feature change or prediction code or training code change, create a new version `saiva-3-day-hosp-v*`
* `/models/saiva-3-day-hosp-v1/Makefile` has the command to create new version

### Training
* After every training a model, use scripts/other/generate_deploy_model.sh to
generate a buildspec.yml which will help in building only required models


### Running Tests

#### Webapp

##### Running tests locally using docker

Run these commands from the webapp directory
```
# docker-compose build
# docker-compose run web python -m pytest
```

##### Running tests in Codebuild using CLI


```
# aws codebuild start-build --project-name WebappTests
```
Navigate to https://console.aws.amazon.com/codesuite/codebuild/570873190097/projects/WebappTests/history?region=us-east-1 to see the test results

##### Running tests in Codebuild using console

Navigate to https://console.aws.amazon.com/codesuite/codebuild/570873190097/projects/WebappTests/history?region=us-east-1 and click on the start build button. By default tests are run on master, but can be overriden from the console.
To view test reports, select a report from the reports tab



