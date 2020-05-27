# function to generate and save music video
import requests
from visualize import song_analysis, generate_images, save_video
from flask import Response


def check_entry(preview, video_id, resolution):

	r = requests.get(preview).status_code

	vis_url = 'http://sample.eba-5jeurmbw.us-east-1.elasticbeanstalk.com/visualize'

	data = {"preview": preview, "video_id": video_id,
			"resolution": resolution}

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
