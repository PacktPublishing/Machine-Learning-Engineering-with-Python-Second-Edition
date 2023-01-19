# INSPIRED BY https://github.com/MartinHeinz/python-project-blueprint/blob/master/Makefile

# Include the other helpful makefile components
include Makefile.help

MODULE := outliers
PROJECT_NAME := mlewp-ed2-ch4-outliers
PYTHON_INTERPRETER := python3
ARCH := $(shell $(PYTHON_INTERPRETER) -c "import platform; print(platform.platform())")
VIRTUALENV := conda
CONDA_EXE ?= ~/anaconda3/bin/conda
EASYDATA_LOCKFILE := environment.$(ARCH).lock.yml

$$(EASYDATA_LOCKFILE): environment.yml
ifeq (conda, $(VIRTUALENV))
	$(CONDA_EXE) env update -n $(PROJECT_NAME) -f $<
	$(CONDA_EXE) env export -n $(PROJECT_NAME) -f $@
	# pip install -e .  # uncomment for conda <= 4.3
else
	$(error Unsupported Environment `$(VIRTUALENV)`. Use conda)
endif

.PHONY: create_environment
## Set up virtual (conda) environment for this project
create_environment: $(EASYDATA_LOCKFILE)
ifeq (conda,$(VIRTUALENV))
	@rm -f $(EASYDATA_LOCKFILE)
	@echo
	@echo "New conda env created. Activate with:"
	@echo ">>> conda activate $(PROJECT_NAME)"
	@echo ">>> make update_environment"
ifneq ("X$(wildcard .post-create-environment.txt)","X")
	@cat .post-create-environment.txt
endif
else
	$(error Unsupported Environment `$(VIRTUALENV)`. Use conda)
endif

.PHONY: delete_environment
## Delete the virtual (conda) environment for this project
delete_environment:
ifeq (conda,$(VIRTUALENV))
	@echo "Deleting conda environment."
	$(CONDA_EXE) env remove -n $(PROJECT_NAME)
	rm -f $(EASYDATA_LOCKFILE)
ifneq ("X$(wildcard .post-delete-environment.txt)","X")
	@cat .post-delete-environment.txt
endif
else
	$(error Unsupported Environment `$(VIRTUALENV)`. Use conda)
endif

.PHONY: update_environment
## Install or update Python Dependencies in the virtual (conda) environment
update_environment: environment_enabled $(EASYDATA_LOCKFILE)
ifneq ("X$(wildcard .post-update-environment.txt)","X")
	@cat .post-update-environment.txt
endif

.PHONY: environment_enabled
# Checks that the conda environment is active
environment_enabled:
ifeq (conda,$(VIRTUALENV))
ifneq ($(notdir ${CONDA_DEFAULT_ENV}), $(PROJECT_NAME))
	$(error Run "$(VIRTUALENV) activate $(PROJECT_NAME)" before proceeding...)
endif
else
	$(error Unsupported Environment `$(VIRTUALENV)`. Use conda)
endif

.PHONY: check_lockfile
# Test that an environment lockfile exists
check_lockfile:
ifeq (X,X$(wildcard $(EASYDATA_LOCKFILE)))
	$(error Run "make update_environment" before proceeding...)
endif

.PHONY: check_environment
## Check if environment is enabled and correctly configured
check_environment: environment_enabled check_lockfile $(EASYDATA_LOCKFILE)

.phony: help_update_easydata
help_update_easydata:
	@echo "\nTo update easydata on an existing repo, verify that you have an 'easydata' branch"
	@echo "\n>>>git rev-parse -q --verify easydata"
	@echo "\nIf no output is given, do this:"
	@echo "\n>>>git branch easydata `git rev-list --max-parents=0 HEAD`"
	@echo "\nIf no output is given, do this:"
	@echo "\nCheck-in all your changes, then merge the new easydata branch into yours"
	@echo "\ngit branch easydata"
	@echo "# replace easydata with https://github.com/hackalog/easydata if needed"
	@echo "pushd .. && cookiecutter --config-file $(PROJECT_NAME)/.easydata.yml easydata -f --no-input && popd"
	@echo "git add -p  # add all the changes"
	@echo "git commit -m 'sync with easydata'"
	@echo "git checkout main"
	@echo "git merge easydata"

.PHONY: debug_environment
## dump useful debugging information to $(DEBUG_FILE)
debug_environment:
	@echo "\n\n======================"
	@echo "\nPlease include the contents $(DEBUG_FILE) when submitting an issue or support request.\n"
	@echo "======================\n\n"
	@echo "##\n## Git status\n##\n" > $(DEBUG_FILE)
	git status >> $(DEBUG_FILE)
	@echo "\n##\n## git log\n##\n" >> $(DEBUG_FILE)
	git log -8 --graph --oneline --decorate --all >> $(DEBUG_FILE)
	@echo "\n##\n## Github remotes\n##\n" >> $(DEBUG_FILE)
	git remote -v >> $(DEBUG_FILE)
	@echo "\n##\n## github SSH credentials\n##\n" >> $(DEBUG_FILE)
	ssh git@github.com 2>&1 | cat >> $(DEBUG_FILE)
	@echo "\n##\n## Conda config\n##\n" >> $(DEBUG_FILE)
	$(CONDA_EXE) config --get >> $(DEBUG_FILE)
	@echo "\n##\n## Conda info\n##\n" >> $(DEBUG_FILE)
	$(CONDA_EXE) info  >> $(DEBUG_FILE)
	@echo "\n##\n## Conda list\n##\n" >> $(DEBUG_FILE)
	$(CONDA_EXE) list >> $(DEBUG_FILE)

.PHONY: unfinished
unfinished:
	@echo "WARNING: this target is unfinished and may be removed or changed dramatically in future releases"

run:
	@python -m $(MODULE)

.PHONY: clean test
clean:
	rm -rf .pytest_cache .coverage .pytest_cache coverage.xml

test:
	@pytest



#lint:
#	@echo "\n${BLUE}Running Pylint against source and test files...${NC}\n"
#	@pylint --rcfile=setup.cfg **/*.py
#	@echo "\n${BLUE}Running Flake8 against source and test files...${NC}\n"
#	@flake8
#	@echo "\n${BLUE}Running Bandit against source files...${NC}\n"
#	@bandit -r --ini setup.cfg

