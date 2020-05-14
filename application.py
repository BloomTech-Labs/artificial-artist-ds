import os
import requests
from flask import Flask, request
from flask_cors import CORS
from flask_s3 import FlaskS3
from visualize import song_analysis, generate_images, save_video

s3 = FlaskS3()

def create_app():
	application = Flask(__name__)
	CORS(application)
	s3.init_app(app)

	@application.route('/')
	def root():
		return "hello world"

	# get user input
	@application.route('/visualize', methods=['GET', 'POST'])
	def visual():
		url = request.args.get('preview')
		video_id = str(request.args.get('video_id'))

		song = requests.get(url)
		open("song.mp3", 'wb').write(song.content)
			
		noise_vectors, class_vectors = song_analysis("song.mp3")

		frames = generate_images(noise_vectors, class_vectors)

		save_video(frames, "song.mp3", video_id)

		return "It worked!"
		#download user input--cache data???
		#feed downloaded song into visualizer
	return application
