import os
import requests
from flask import Flask
from flask import request
from pydub import AudioSegment
from visualize import song_analysis, generate_images, save_video

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
	open("song.mp3", 'wb').write(song.content)
		
	noise_vectors, class_vectors = song_analysis("song.mp3")

	frames = generate_images(noise_vectors, class_vectors)

	save_video(frames, "song.mp3")

	return "unit-test"

	#download user input--cache data???
	#feed downloaded song into visualizer

if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()	


