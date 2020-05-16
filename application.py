import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from visualize import song_analysis, generate_images, save_video

def create_app():
	application = Flask(__name__)
	CORS(application)

	@application.route('/')
	def root():
		return "hello world"

	@application.route('/visualize', methods=['GET', 'POST'])
	def visual():
		url = request.args.get('preview')
		video_id = str(request.args.get('video_id'))

		song = requests.get(url)
		open("song.mp3", 'wb').write(song.content)
			
		noise_vectors, class_vectors = song_analysis("song.mp3")

		frames = generate_images(noise_vectors, class_vectors)

		s3_url = save_video(frames, "song.mp3", video_id)

		return jsonify(video_url=s3_url, video_id=video_id)

	return application
