import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import check_entry, generate_and_save


def create_app():
	application = Flask(__name__)
	CORS(application)

	@application.route('/')
	def root():
		return "Index; nothing to see here."

	@application.route('/entry', methods=['POST'])
	def check_url():

		reqs = request.get_json()['params']

		preview = reqs['preview']
		video_id = reqs['video_id']

		# default values initiated
		resolution = '128'
		im_group = None
		jitter = 0.5
		depth = 1
		truncation = 0.5
		pitch_sensitivity = 220
		tempo_sensitivity = 0.25
		smooth_factor = 20

		# checks to see if other values are specified by user
		if 'resolution' in reqs:
			resolution = reqs['resolution']

		if 'im_group' in reqs:
			im_group = reqs['im_group']

		if 'jitter' in reqs:
			jitter = reqs['jitter']

		if 'depth' in reqs:
			depth = reqs['depth']

		if 'truncation' in reqs:
			truncation = reqs['truncation']

		if 'pitch_sensitivity' in reqs:
			pitch_sensitivity = reqs['pitch_sensitivity']

		if 'tempo_sensitivity' in reqs:
			tempo_sensitivity = reqs['tempo_sensitivity']

		if 'smooth_factor' in reqs:
			smooth_factor = reqs['smooth_factor']

		return check_entry(preview, video_id, resolution, im_group, jitter,
							depth, truncation, pitch_sensitivity,
							tempo_sensitivity, smooth_factor)

	@application.route('/visualize', methods=['POST'])
	def visual():

		reqs = request.get_json()

		preview = reqs['preview']
		video_id = str(reqs['video_id'])
		resolution = str(reqs['resolution'])
		classes = reqs['classes']
		jitter = reqs['jitter']
		depth = reqs['depth']
		truncation = reqs['truncation']
		pitch_sensitivity = reqs['pitch_sensitivity']
		tempo_sensitivity = reqs['tempo_sensitivity']
		smooth_factor = reqs['smooth_factor']

		return generate_and_save(preview, video_id, resolution, classes,
								 jitter, depth, truncation,
								 pitch_sensitivity, tempo_sensitivity,
								 smooth_factor)

	return application
