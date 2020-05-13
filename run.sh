#!/bin/sh
source venv/bin/activate
exec gunicorn -w 8 -t 4 -b 127.0.0.1:5000 --access-logfile - --error-logfile - run:application