from flask import Flask
import requests

def create_app():
	app = Flask(__name__)

	songinfo = requests.args()

	@app.route('/')
	def root():
		return "hello world"

# get user input
	@app.route('/visualize')
		url = songinfo['data'][0]['preview']
		artist = songinfo['data'][0]['artist']
		title = songinfo['data'][0]['title_short']

		song = requests.get(url)
		open('song.mp3', 'wb').write(song.content)



#download user input--cache data???
#feed downloaded song into visualizer
	return app