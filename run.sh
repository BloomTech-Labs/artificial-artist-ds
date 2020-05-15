#!/bin/sh
source venv/bin/activate
exec gunicorn -w 2 -t 3600 -b 0.0.0.0:5000 --access-logfile - --error-logfile - run:application