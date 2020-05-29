import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import check_entry, generate_and_save


def create_app():
	application = Flask(__name__)
	CORS(application)

	@application.route('/')
	def root():
		return "Index; nothing to see here."

	@application.route('/entry', methods=['GET', 'POST'])
	def check_url():

		if request.method == 'POST':
			reqs = request.get_json()

			preview = (reqs['preview'])
			video_id = (reqs['video_id'])
			resolution = '128'
			if 'resolution' in reqs:
				resolution = (reqs['resolution'])
			im_group = None
			if 'im_group' in reqs:
				im_group = (reqs['resolution'])

			return check_entry(preview, video_id, resolution, im_group)

		preview = str(request.args.get('preview'))
		video_id = str(request.args.get('video_id'))
		im_group = request.args.get('im_group')
		if im_group != None:
			im_group = str(im_group)
		resolution = request.args.get('resolution')
		if resolution == None:
			resolution = '128'
		

		return check_entry(preview, video_id, str(resolution), im_group)

	@application.route('/visualize', methods=['GET', 'POST'])
	def visual():

		if request.method == 'POST':
			reqs = request.get_json()

			preview = reqs['preview']
			resolution = reqs['resolution']
			video_id = reqs['video_id']
			classes = reqs['classes']

			return generate_and_save(preview, video_id, resolution, classes)

		preview = str(request.args.get('preview'))
		video_id = str(request.args.get('video_id'))
		resolution = str(request.args.get('resolution'))

		return generate_and_save(preview, video_id, resolution)

	return application