all: dev

# For example, to build and run the docker container locally in development mode, just run:
# $ make dev

# or to run a command within the container run:
# $ make run command='python -c "print(1)"'

name ?= saiva-3-day-hosp-v3
version ?= dev
registry = dkr.ecr.us-east-1.amazonaws.com
account ?= 570873190097
aws_region ?= us-east-1
profile_name ?= default

SSH_KEY_PATH = ~/.ssh/id_rsa
BRANCH_NAME ?= $(shell git rev-parse --abbrev-ref HEAD)

image:
	cp -r -p $(SSH_KEY_PATH) ./id_rsa
	echo "copy static folder.........."
	\cp -r ../../static ./src/
	echo "Building docker image..."
	docker build -t ${name} .
	rm ./id_rsa

image_dev:
	cp -r -p $(SSH_KEY_PATH) ./id_rsa
	echo "copy static folder.........."
	\cp -r ../../static ./src/
	echo "Building docker image..."
	docker build -t ${name}-local-dev -f Dockerfile.local .
	rm ./id_rsa

dev: image_dev
	echo "Running Docker image locally in development mode."
	docker run --name ${name}-local-dev -d --rm -p 8443 -p 8888 -p 5000 -v ~/.aws:/root/.aws:ro -v $(shell pwd)/data:/data -v $(shell pwd)/src:/src -v $(shell pwd)/vscode-user-data:/vscode-user-data ${name}-local-dev
	
	echo "Docker image running - information on available ports below"
	docker ps

run: image
	@echo "Running Docker image with specified command."
	docker run -i -t --rm -p 8443 -p 8888 -v ~/.aws:/root/.aws:ro -v $(shell pwd)/data:/data -v $(shell pwd)/src:/src ${name} $(command)

# Used to create versions
# make copy_code ARGS="saiva-3-day-hosp-v3"
copy_code:
	@echo $(ARGS)
	rm -rf $(ARGS)
	cp -R ./ ../$(ARGS)/

# Updates dependencies and regenerates Pipfile.lock
lock:
	pipenv lock

# Do a AWS codeguru security scan
scan:
	sh .circleci/run_codeguru_security.sh saiva-ml-$(BRANCH_NAME) . us-east-1

image-x86:
	cp -r -p $(SSH_KEY_PATH) ./id_rsa
	echo "copy static folder.........."
	\cp -r ../../static ./src/
	echo "Building docker image..."
	docker buildx build --platform linux/amd64 -t ${name}:${version} .
	rm ./id_rsa

deploy: image-x86
	echo $(shell aws ecr get-login-password --region ${aws_region} --profile ${profile_name}) | docker login -u AWS --password-stdin ${account}.${registry}
	docker tag ${name}:${version} ${account}.${registry}/${name}:${version}
	docker push ${account}.${registry}/${name}:${version}