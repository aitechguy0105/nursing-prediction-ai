all: dev

# This makefile contains some convenience commands for deploying and publishing.

# For example, to build and run the docker container locally in development mode, just run:
# $ make

# or to publish the :latest version to the specified registry as :1.0.0, run:
# $ make publish version=1.0.0

# or to run a command within the container run:
# $ make run command='python -c "print(1)"'

name = $(shell basename $(shell pwd))
registry = dkr.ecr.us-east-1.amazonaws.com
version ?= latest
account ?= 570873190097
s3bucket ?= production-s3-18fbe7wpqplsk-databucket-1fwcegwxmbigs
modelid ?= 82af079850514a92a5bc10d7294bd9d2

image:
	echo "Building docker image..."
	docker build -t ${name}:${version} .

dev: image
	echo "Running Docker image locally in development mode."
	docker run --name ${name} -d --rm -p 8443:8443 -p 8888:8888 -v ~/.aws:/root/.aws:ro -v $(shell pwd)/vscode-user-data:/vscode-user-data -v $(shell pwd)/data:/data -v $(shell pwd)/code:/code -v $(shell pwd)/notebooks:/notebooks ${name}:${version}
	
	echo "Docker image running - information on available ports below"
	docker ps

publish: image
	echo "Publishing Docker image to registry..."
	$(shell aws ecr get-login --no-include-email --region us-east-1)
	docker tag ${name}:${version} ${account}.${registry}/${name}:${version}
	docker push ${account}.${registry}/${name}:${version}

run: image
	echo "Running Docker image with specified command."
	docker run -i -t --rm -p 8443:8443 -p 8888:8888 -v ~/.aws:/root/.aws:ro -v $(shell pwd)/data:/data -v $(shell pwd)/code:/code ${name}:${version} $(command)

getmodel:
	aws s3 sync s3://${s3bucket}/models/${modelid} ./data/model
