import os
import requests
from flask import Flask
from flask import request
from flask_cors import CORS
from visualize import song_analysis, generate_images, save_video

def create_app():
	application = Flask(__name__)
	CORS(application)

	@application.route('/')
	def root():
		return "hello world"

	# get user input
	@application.route('/visualize', methods=['GET', 'POST'])
	def visual():
		url = request.args.get('preview')
		artist = request.args.get('artist')
		title = request.args.get('title_short')

		song = requests.get(url)
		open("song.mp3", 'wb').write(song.content)
			
		noise_vectors, class_vectors = song_analysis("song.mp3")

		frames = generate_images(noise_vectors, class_vectors)

		save_video(frames, "song.mp3")

		return 'It worked!'
		#download user input--cache data???
		#feed downloaded song into visualizer
	return application
