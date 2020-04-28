from flask import Flask
from flask import request
import requests
from .Model import visualize


application = Flask(__name__)

@application.route('/')
def root():
	return "hello world"

# get user input
@application.route('/visualize')
def visual():
	url = request.args.get('preview')
	artist = request.args.get('artist')
	title = request.args.get('title_short')

	song = requests.get(url)
	open('song.mp3', 'wb').write(song.content)

	return "unit-test"

	#download user input--cache data???
	#feed downloaded song into visualizer

if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()	


