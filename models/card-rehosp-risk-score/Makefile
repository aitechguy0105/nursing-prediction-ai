all: dev

# For example, to build and run the docker container locally in development mode, just run:
# $ make dev

# or to run a command within the container run:
# $ make run command='python -c "print(1)"'


name ?= $(shell basename $(shell pwd))

image:
	echo "Building docker image..."
	docker build -t ${name} .

dev: image
	echo "Running Docker image locally in development mode."
	docker run --name ${name} -d --rm -p 8443 -p 8888 -p 5000 -v ~/.aws:/root/.aws:ro -v $(shell pwd)/data:/data -v $(shell pwd)/src:/src -v $(shell pwd)/vscode-user-data:/vscode-user-data ${name}
	
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
