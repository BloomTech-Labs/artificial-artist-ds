# function to generate and save music video
import os
from os.path import join
import shutil
import requests
import random
import boto3
from botocore.exceptions import ClientError
import logging
from flask import Response
import moviepy.editor as mpy
from PIL import Image
from config import *
from image_groups import IMAGE_GROUPS
from visualize import song_analysis, generate_images


def choose_classes(im_group):
	if im_group == None:
		im_group = random.sample(IMAGE_GROUPS.keys(), 1)[0]

	if len(IMAGE_GROUPS[im_group]) < 4:
		im_classes = IMAGE_GROUPS[im_group]
	else:
		im_classes = random.sample(IMAGE_GROUPS[im_group], 4)

	im_classes.append(random.sample(list(range(1000)), 2))
	
	return im_classes


def check_entry(preview, video_id, resolution, im_group, jitter,
				depth, truncation, pitch_sensitivity, tempo_sensitivity,
				smooth_factor):

	r = requests.get(preview).status_code

	vis_url = 'http://artificial-artist.eba-cyfpphb2.us-east-1.elasticbeanstalk.com/visualize'

	classes = choose_classes(im_group)

	data = {"preview": preview, "video_id": video_id, "resolution": resolution,
			"classes": classes, "jitter": jitter, "depth": depth,
			"truncation": truncation, "pitch_sensitivity": pitch_sensitivity,
			"tempo_sensitivity": tempo_sensitivity, "smooth_factor": smooth_factor}

	if r == 200:
		try:
			requests.post(vis_url, json=data, timeout=3)
		except:
			pass
		return Response('Accepted', status=202, mimetype='application/json')

	else:
		return Response(f"{str(vis_url)} not found.", status=404,
						mimetype='application/json')


def upload_file_to_s3(mp4file, jpgfile, bucket_name=S3_BUCKET, acl="public-read"):
	"""
	Saves mp4 and jpg of created video to S3 Bucket

	inputs:
			mp4file: STR; location of mp4
			jpgfile: STR; location of jpgfile
			bucket_name: STR; S3 bucket name
			acl: STR; allows certain security settings for file access

	"""

	S3_LOCATION = 'https://{}.s3.amazonaws.com/'.format(bucket_name)

	s3 = boto3.client(
		"s3",
		aws_access_key_id=S3_KEY,
		aws_secret_access_key=S3_SECRET
	)

	try:
		with open(mp4file, "rb") as f:
			s3.upload_fileobj(
				f,
				bucket_name,
				mp4file,
				ExtraArgs={
					"ACL": acl,
					"ContentType": "video/mp4"
				}
			)

		with open(jpgfile, "rb") as f:
			s3.upload_fileobj(
				f,
				bucket_name,
				jpgfile,
				ExtraArgs={
					"ACL": acl,
					"ContentType": "image/jpg"
				}
			)

	except ClientError as e:
		logging.error(e)
		return "error uploading"

	return 'Succesfully uploaded files to S3'


def save_video(tmp_folder_path, song, outname, frame_length = 512):
	"""
	Input: 
			tmp_folder_path: STR; path of folder containing images for frames
			song: STR; location of song
			outname: STR; desired name of files

	Output: 
			creates video in mp4 format, and jpg for thumbnail and calls to save

	"""
	files_path = [os.path.join(tmp_folder_path, x)
					for x in os.listdir(tmp_folder_path) if x.endswith('.png')]

	files_path.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	aud = mpy.AudioFileClip(song, fps=44100)
	aud.duration = 30

	# creates mp4
	clip = mpy.ImageSequenceClip(files_path, fps=22050 / frame_length)
	clip = clip.set_audio(aud)
	clip.write_videofile(outname + ".mp4", audio_codec='aac')

	# saves thumbnail
	thumbnail = Image.open(files_path[-1])
	thumbnail.save(outname + ".jpg")

	# cleans temp directory
	if os.path.exists(tmp_folder_path):
		shutil.rmtree(tmp_folder_path)

	return upload_file_to_s3(outname + ".mp4", outname + ".jpg")


def generate_and_save(preview, video_id, resolution, classes, jitter, depth, 
						truncation, pitch_sensitivity, tempo_sensitivity,
						smooth_factor):

	song = requests.get(preview)

	song_name = f"{video_id}.mp3"

	open(song_name, 'wb').write(song.content)

	noise_vectors, class_vectors = song_analysis(song_name, classes,
												 jitter, depth, truncation,
												 pitch_sensitivity,
												 tempo_sensitivity, 
												 smooth_factor)

	tmp_folder_path = generate_images(video_id, noise_vectors, class_vectors, 
										resolution,truncation)

	return save_video(tmp_folder_path, song_name, video_id)



