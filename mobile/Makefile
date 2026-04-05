SHELL := /bin/bash
# Color Ref: https://linuxhandbook.com/change-echo-output-color/
R='\033[0;31m'   #'0;31' is Red's ANSI color code
G='\033[0;32m'   #'0;32' is Green's ANSI color code
Y='\033[1;32m'   #'1;32' is Yellow's ANSI color code
B='\033[0;34m'   #'0;34' is Blue's ANSI color code
NOCOLOR='\033[0m'

PUBSPEC_FILES := $(shell find . -name 'pubspec.yaml')
MAKE_FILES := $(shell find . -mindepth 1 -name 'Makefile')

init: get gen
	cd app/ios && pod update && pod install
	@echo "${G}Finished Project Setup${NOCOLOR}"

gen:
	@for file in $(PUBSPEC_FILES); do \
		cd $${file%pubspec.yaml} && \
		pwd && \
		dart run build_runner build --delete-conflicting-outputs && \
		cd "$$OLDPWD"; \
	done
	@echo "${G}Build runner completed successfully${NOCOLOR}"

get:
	@for file in $(PUBSPEC_FILES); do \
		echo "Processing $$file"; \
		cd $${file%pubspec.yaml} && \
		flutter pub get && \
		cd "$$OLDPWD"; \
	done
	@echo "${G}Flutter pub get completed successfully${NOCOLOR}"

clean:
	@for file in $(PUBSPEC_FILES); do \
		cd $${file%pubspec.yaml} && \
		pwd && \
		flutter clean && \
		cd "$$OLDPWD"; \
	done
	@echo "${G}Flutter clean completed successfully${NOCOLOR}"
