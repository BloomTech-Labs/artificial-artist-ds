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

			preview = reqs['preview']
			video_id = reqs['video_id']
			
			#default values initiated
			resolution = '128'
			im_group = None
			jitter = 0.5
			depth = 1
			truncation = 0.5
			pitch_sensitivity = 220
			tempo_sensitivity = 0.25

			#checks to see if other values are specified by user
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

			return check_entry(preview, video_id, resolution, im_group, jitter, 
								depth, truncation, pitch_sensitivity, 
								tempo_sensitivity)

		preview = str(request.args.get('preview'))
		video_id = str(request.args.get('video_id'))
		
		resolution = request.args.get('resolution')
		if resolution != None:
			resolution = str(resolution)
		
		im_group = request.args.get('im_group')
		if im_group != None:
			im_group = str(im_group)
		
		jitter = request.args.get('jitter')
		if jitter == None:
			jitter = 0.5
		
		depth = request.args.get('depth')
		if depth == None:
			depth = 1
		
		truncation = request.args.get('truncation')
		if truncation == None:
			truncation = 0.5

		pitch_sensitivity = request.args.get('pitch_sensitivity')
		if pitch_sensitivity == None:
			pitch_sensitivity = 220

		tempo_sensitivity = request.args.get('tempo_sensitivity')
		if tempo_sensitivity == None:
			tempo_sensitivity = 0.25
		
		return check_entry(preview, video_id, resolution, im_group, jitter, 
								depth, truncation, pitch_sensitivity, 
								tempo_sensitivity)

	@application.route('/visualize', methods=['GET', 'POST'])
	def visual():

		if request.method == 'POST':
			reqs = request.get_json()

			preview = reqs['preview']
			video_id = reqs['video_id']
			resolution = reqs['resolution']
			classes = reqs['classes']
			jitter = reqs['jitter']
			depth = reqs['depth']
			truncation = reqs['truncation']
			pitch_sensitivity = reqs['pitch_sensitivity']
			tempo_sensitivity = reqs['tempo_sensitivity']

			return generate_and_save(preview, video_id, resolution, classes,
										jitter, depth, truncation, 
										pitch_sensitivity, tempo_sensitivity)

		preview = str(request.args.get('preview'))
		video_id = str(request.args.get('video_id'))
		resolution = str(request.args.get('resolution'))
		classes = str(request.args.get('classes'))
		jitter = request.args.get('jitter')
		depth = request.args.get('depth')
		truncation = request.args.get('truncation')
		pitch_sensitivity = request.args.get('pitch_sensitivity')
		tempo_sensitivity = request.args.get('tempo_sensitivity')

		return generate_and_save(preview, video_id, resolution, classes,
										jitter, depth, truncation, 
										pitch_sensitivity, tempo_sensitivity)

	return application