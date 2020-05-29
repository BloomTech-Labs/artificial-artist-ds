# function to generate and save music video
import requests
from visualize import song_analysis, generate_images, save_video
from flask import Response
from image_groups import IMAGE_GROUPS
import random

def choose_classes(im_group):
	if im_group == None:
		im_group = random.sample(IMAGE_GROUPS.keys(), 1)

	if len(IMAGE_GROUPS[im_group[0]]) < 4:
		im_classes = IMAGE_GROUPS[im_group[0]]
	else:
		im_classes = random.sample(IMAGE_GROUPS[im_group[0]], 4)
	return im_classes

def check_entry(preview, video_id, resolution, im_group):

	r = requests.get(preview).status_code

	vis_url = 'http://sample.eba-5jeurmbw.us-east-1.elasticbeanstalk.com/visualize'

	classes = choose_classes(im_group)

	data = {"preview": preview, "video_id": video_id,
			"resolution": resolution, "classes": classes}

	if r == 200:
		try:
			requests.post(vis_url, json=data, timeout=3)
		except:
			pass
		return Response('Accepted', status=202, mimetype='application/json')

	else:
		return Response(f"{str(url)} not found.", status=404,
						mimetype='application/json')

def generate_and_save(preview, video_id, resolution):

	song = requests.get(preview)

	open("song.mp3", 'wb').write(song.content)

	noise_vectors, class_vectors = song_analysis("song.mp3")

	frames = generate_images(noise_vectors, class_vectors, resolution)

	save_video(frames, "song.mp3", video_id)

	return 'Saved to S3'
