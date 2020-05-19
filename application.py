import os
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from visualize import song_analysis, generate_images, save_video

def create_app():
	application = Flask(__name__)
	CORS(application)

	@application.route('/')
	def root():
		return "Index; nothing to see here."


	@application.route('/entry', methods=['GET','POST'])
	def check_url():

		url = request.args.get('preview')
		video_id = str(request.args.get('video_id'))

		r = requests.get(url).status_code

		
		if r == 200:
			try:
				requests.get(f"http://sample.eba-5jeurmbw.us-east-1.elasticbeanstalk.com/visualize?preview={url}&video_id={video_id}", timeout=3)
			except:
				pass
			return Response('Accepted', status=202, mimetype='application/json')
		
		else:
			return Response(url, status=404, mimetype='application/json')

			
	@application.route('/visualize', methods=['GET','POST'])	
	def visual():

		url = request.args.get('preview')
		video_id = str(request.args.get('video_id'))

		song = requests.get(url)
		open("song.mp3", 'wb').write(song.content)
			
		noise_vectors, class_vectors = song_analysis("song.mp3")

		frames = generate_images(noise_vectors, class_vectors)

		s3_url = save_video(frames, "song.mp3", video_id)

		backend = f"http://artificialartistbe-env.eba-avxhjd7c.us-east-1.elasticbeanstalk.com/api/videos/{video_id}"

		data = {"location": s3_url}

		return requests.put(backend, json = data)

	return application
