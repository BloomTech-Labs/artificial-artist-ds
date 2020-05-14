#!/bin/sh
source venv/bin/activate
exec gunicorn -w 4 -t 420 -b 0.0.0.0:5000 --access-logfile - --error-logfile - run:application