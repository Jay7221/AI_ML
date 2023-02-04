#! /bin/bash

read FILE
touch $FILE.py
echo FILE=$FILE.py > .env
code $FILE.py
