.DEFAULT_GOAL := help
TAG_TRAINING := 427750820708.dkr.ecr.us-east-1.amazonaws.com/maml_gdr2

build-train:  ## Build the Docker image to train on AWS
	docker build \
	    --secret id=github_token,src=.secrets/github_token \
		--secret id=wandb_api_key,src=.secrets/wandb_api_key \
		--secret id=hf_token,src=.secrets/huggingface_token \
		--build-arg HF_TOKEN=$$(cat .secrets/huggingface_token | tr -d '\n') \
	    --platform linux/amd64 \
	    --file Dockerfile \
	    --tag ${TAG_TRAINING} \
	    .

push-train:  ## Push the training Docker image to AWS; must build it first
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 427750820708.dkr.ecr.us-east-1.amazonaws.com
	docker push ${TAG_TRAINING}
	docker image prune -f


build-and-push-train: build-train push-train  ## First build then push the training image

check-lint:  ## Check if the files are linted properly
	autoflake -r --quiet --check-diff --remove-all-unused-imports --ignore-init-module-imports --remove-duplicate-keys --remove-unused-variables .
	black -l 120 --diff --color .
	isort -c --profile black -l 120 .

lint:  ## Lint files
	autoflake -r --quiet --in-place --remove-all-unused-imports --ignore-init-module-imports --remove-duplicate-keys --remove-unused-variables .
	black -l 120 .
	isort --profile black -l 120 .

help:  ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-25s %s\n" "task" "help" ; \
	printf "%-25s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-25s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
