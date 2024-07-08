all: dev

# For example, to build and run the docker container locally in development mode, just run:
# $ make dev

# or to run a command within the container run:
# $ make run command='python -c "print(1)"'

name ?= saiva-3-day-hosp-v6
version ?= dev
registry = dkr.ecr.us-east-1.amazonaws.com
account ?= 570873190097
aws_region ?= us-east-1
profile_name ?= default

SSH_KEY_PATH = ~/.ssh/id_rsa

# automatic training
automatic_training_name ?= automatic-training
model_target_path = .
automatic_training_target_path = src/training_pipeline
branch ?= $(shell git rev-parse --abbrev-ref HEAD)
env ?= dev
datacards_image_env ?= prod

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

# AUTOMATIC TRAINING
build_native:
	cp -r -p $(SSH_KEY_PATH) ./id_rsa
	@echo "Building docker image (native)"
	docker build -f src/training_pipeline/Dockerfile -t ${automatic_training_name}:${branch} --build-arg MODEL_TARGET_PATH=${model_target_path} --build-arg AUTOMATIC_TRAINING_TARGET_PATH=${automatic_training_target_path} .

build_ecr: build_native
	@echo "Pushing image to ECR"
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 570873190097.dkr.ecr.us-east-1.amazonaws.com
	docker tag ${automatic_training_name}:${branch} 570873190097.dkr.ecr.us-east-1.amazonaws.com/saiva-automatic-training:${branch}
	docker push 570873190097.dkr.ecr.us-east-1.amazonaws.com/saiva-automatic-training:${branch}

# AUTOMATIC TRAINING - locally

build_native_local:
	cp -r -p $(SSH_KEY_PATH) ./id_rsa
	@echo "Building docker image (native)"
	docker build -f src/training_pipeline/Dockerfile.local -t ${automatic_training_name}:${branch} --build-arg MODEL_TARGET_PATH=${model_target_path} --build-arg AUTOMATIC_TRAINING_TARGET_PATH=${automatic_training_target_path} .
	rm ./id_rsa

build_amd64_local:
	cp -r -p $(SSH_KEY_PATH) ./id_rsa
	@echo "Building docker image (linux/amd64)"
	docker build -f src/training_pipeline/Dockerfile.local -t ${automatic_training_name}:${branch} --build-arg MODEL_TARGET_PATH=${model_target_path} --build-arg AUTOMATIC_TRAINING_TARGET_PATH=${automatic_training_target_path} --platform linux/amd64 .
	rm ./id_rsa

run_automatic_training_local: 
	sudo apt install -y jq
	./automated_training_local.sh env=$(env) datacards_image_env=$(datacards_image_env) command=$(command) step=$(step) start_step=$(start_step)

run_automatic_training_amd64_local:
	sudo apt install -y jq
	./automated_training_local.sh env=$(env) datacards_image_env=$(datacards_image_env) command=$(command) step=$(step) start_step=$(start_step) use_amd64=true


run_automatic_training_local_script: build_native_local
	@echo "Running Docker image with specified command"
	docker run -i -t --rm -v ~/.aws:/root/.aws:ro -v $(shell pwd)/data:/data -v $(shell pwd)/src:/src ${automatic_training_name}:${branch} $(command)


run_automatic_training_amd64_local_script: build_amd64_local
	@echo "Running Docker image with specified command"
	docker run -i -t --rm -v ~/.aws:/root/.aws:ro -v $(shell pwd)/data:/data -v $(shell pwd)/src:/src ${automatic_training_name}:${branch} $(command)