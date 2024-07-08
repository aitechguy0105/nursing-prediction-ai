all: dev

# For example, to build and run the docker container locally in development mode, just run:
# $ make dev

# or to publish the :latest version to the specified registry as :1.0.0, run:
# $ make publish version=dev

# or to run a command within the container run:
# $ make run command='python -c "print(1)"'


name ?= $(shell basename $(shell pwd))
registry = dkr.ecr.us-east-1.amazonaws.com
version ?= dev
account ?= 570873190097
full_version = ${version}_$(shell git tag -l build_* --sort=committerdate | tail -1)

image:
	echo "Building docker image - ${name}:${version}"
	docker build -t ${name}:${version} .

dev: image
	echo "Running Docker image locally in development mode."
	docker run --name ${name} -d --rm -p 8443 -p 8888 -p 5000 -v ~/.aws:/root/.aws:ro -v $(shell pwd)/data:/data -v $(shell pwd)/src:/src -v $(shell pwd)/vscode-user-data:/vscode-user-data ${name}:${version}

	echo "Docker image running - information on available ports below"
	docker ps

# Updates dependencies and regenerates Pipfile.lock
lock:
	pipenv lock

