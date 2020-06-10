import librosa
import numpy as np
import moviepy.editor as mpy
import random
import torch
from scipy.misc import toimage, imsave
from tqdm import tqdm
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
									   save_as_images, display_in_terminal)
import boto3
from botocore.exceptions import ClientError
import logging
from config import *
import os
from os.path import isfile, join
import shutil


def model_resolution(resolution):
	"""
	set model's resolution, default 128
	128, 256, or 512 
	lower = faster generation, lower quality.

	"""
	model_name = 'biggan-deep-' + resolution
	model = BigGAN.from_pretrained(model_name)

	return model


def song_duration(duration=30):
	"""
	Song duration in seconds, returns fram_lim
	default = 30 seconds

	"""
	seconds = duration
	frame_lim = int(np.floor(seconds * 22050 / frame_length / batch_size))

	return frame_lim


# set pitch sensitivity
def sensitivity_pitch(pitch_sensitivity):
	"""
	INT
	Set how quickly images move according to pitch
	Default 220
	Recommended range: 200 – 295

	"""
	pitch_sensitivity = (300 - pitch_sensitivity) * 512 / frame_length

	return pitch_sensitivity


# set tempo sensitivity
def sensitivity_tempo(tempo_sensitivity):
	"""
	FLOAT between 0 and 1
	Set how quickly images morph due to tempo
	Default 0.25
	Recommended range: 0.05 – 0.8

	"""
	tempo_sensitivity = tempo_sensitivity * frame_length / 512

	return tempo_sensitivity


# can reduce this number to make clearer images or increase to reduce computational load
# default: 512
# range: multiples of 64
frame_length = 512

# BigGAN generates the images in batches of size [batch_size].
# default 32
# only reason to lower this is if you run out of cuda memory. will take
# slightly longer.
batch_size = 32

# set device
# use cuda or face a generation time in the hours. You have been warned.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def smooth_rate(smooth_factor):
	"""
	int > 0
	smooths the class vectors to prevent small fluctuations in pitch from causing the frames to go back and forth
	default 20
	recommended range: 10 – 30

	"""
	if smooth_factor > 1:
		smooth_factor = int(smooth_factor * 512 / frame_length)
	else:
		smooth_factor = smooth_factor

	return smooth_factor


def new_jitters(jitter):
	"""
	update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity

	"""
	jitters = np.zeros(128)

	for j in range(128):
		if random.uniform(0, 1) < 0.5:
			jitters[j] = 1
		else:
			jitters[j] = 1 - jitter

	return jitters


def new_update_dir(nv2, update_dir, truncation, tempo_sensitivity):
	"""
	changes the direction of the noise vector

	"""
	for ni, n in enumerate(nv2):
		if n >= 2 * truncation - sensitivity_tempo(tempo_sensitivity):
			update_dir[ni] = -1

		elif n < -2 * truncation + sensitivity_tempo(tempo_sensitivity):
			update_dir[ni] = 1

	return update_dir


#
def smooth(class_vectors, smooth_factor):
	"""
	smooth class vectors

	"""
	if smooth_factor == 1:
		return class_vectors

	class_vectors_terp = []

	for c in range(int(np.floor(len(class_vectors) / smooth_factor) - 1)):
		ci = c * smooth_factor
		cva = np.mean(class_vectors[int(ci):int(ci) + smooth_factor], axis=0)
		cvb = np.mean(
			class_vectors[int(ci) + smooth_factor:int(ci) + smooth_factor * 2], axis=0)

		for j in range(smooth_factor):
			cvc = cva * (1 - j / (smooth_factor - 1)) + \
				cvb * (j / (smooth_factor - 1))
			class_vectors_terp.append(cvc)

	return np.array(class_vectors_terp)


def normalize_cv(cv2):
	"""
	normalize class vector between 0-1

	"""
	min_class_val = min(i for i in cv2 if i != 0)
	for ci, c in enumerate(cv2):
		if c == 0:
			cv2[ci] = min_class_val
	cv2 = (cv2 - min_class_val) / np.ptp(cv2)

	return cv2


def song_analysis(song, classes, jitter, depth, truncation,pitch_sensitivity, 
					tempo_sensitivity, smooth_factor):
	"""
	creates the class and noise vectors files

	Inputs:
			song: STR; path of 30 second mp3 file
			classes: LIST; classes by index from ImageNet1000 -max 12 classes
			jitter: FLOAT 0 to 1
			depth: FLOAT 0 to 1
			truncation: FLOAT 0 to 1
			pitch_sensitivity: INT 1-299
			tempo_sensitivity: FLOAT 0 to 1
			smooth_factor: INT > 0
	Output:
			noise and class vectors of song based on input variables

	"""
	# read song: audio waveform and sampling rate saved
	# y = time, sr = sample rate
	y, sr = librosa.load(song)

	# create spectrogram
	spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000,
											hop_length=frame_length)

	# get mean power at each time point
	specm = np.mean(spec, axis=0)

	# compute power gradient across time points
	gradm = np.gradient(specm)

	# set max to 1
	gradm = gradm / np.max(gradm)

	# set negative gradient time points to zero
	gradm = gradm.clip(min=0)

	# normalize mean power between 0-1
	specm = (specm - np.min(specm)) / np.ptp(specm)

	# create chromagram of pitches X time points
	chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)

	# sort pitches by overall power
	chromasort = np.argsort(np.mean(chroma, axis=1))[::-1]

	# gets # of classes
	num_classes = len(classes)

	# initialize first class vector
	cv1 = np.zeros(1000)

	for pi, p in enumerate(chromasort[:num_classes]):
		if num_classes < 12:
			cv1[classes[pi]] = chroma[p][
				np.min([np.where(chrow > 0)[0][0] for chrow in chroma])]
		else:
			cv1[classes[p]] = chroma[p][
				np.min([np.where(chrow > 0)[0][0] for chrow in chroma])]

	# initialize first noise vector
	nv1 = truncated_noise_sample(truncation=truncation)[0]

	# initialize list of class and noise vectors
	class_vectors = [cv1]
	noise_vectors = [nv1]

	# initialize previous vectors (will be used to track the previous frame)
	cvlast = cv1
	nvlast = nv1

	# initialize the direction of noise vector unit updates
	update_dir = np.zeros(128)

	for ni, n in enumerate(nv1):
		if n < 0:
			update_dir[ni] = 1
		else:
			update_dir[ni] = -1

	# initialize noise unit update
	update_last = np.zeros(128)

	for i in tqdm(range(len(gradm))):

		# print progress
		pass

		if i % 200 == 0:
			jitters = new_jitters(jitter)

		# get last noise vector
		nv1 = nvlast

		# set noise vector update based on direction, sensitivity, jitter, and
		# combination of overall power and gradient of power
		update = np.array([sensitivity_tempo(tempo_sensitivity) for k in range(
			128)]) * (gradm[i] + specm[i]) * update_dir * jitters

		# smooth the update with the previous update (to avoid overly sharp
		# frame transitions)
		update = (update + update_last * 3) / 4

		# set last update
		update_last = update

		# update noise vector
		nv2 = nv1 + update

		# append to noise vectors
		noise_vectors.append(nv2)

		# set last noise vector
		nvlast = nv2

		# update the direction of noise units
		update_dir = new_update_dir(nv2, update_dir, truncation, 
									tempo_sensitivity)

		# get last class vector
		cv1 = cvlast

		# generate new class vector
		cv2 = np.zeros(1000)

		for j in range(num_classes):
			cv2[classes[j]] = (cvlast[classes[j]] +
								((chroma[chromasort[j]][i]) / (sensitivity_pitch(pitch_sensitivity)))) / (1 + (1 / ((sensitivity_pitch(pitch_sensitivity)))))

		# if more than 6 classes, normalize new class vector between 0 and 1,
		# else simply set max class val to 1
		if num_classes > 6:
			cv2 = normalize_cv(cv2)
		else:
			cv2 = cv2 / np.max(cv2)

		# adjust depth
		cv2 = cv2 * depth

		# this prevents rare bugs where all classes are the same value
		if np.std(cv2[np.where(cv2 != 0)]) < 0.0000001:
			cv2[classes[0]] = cv2[classes[0]] + 0.01

		# append new class vector
		class_vectors.append(cv2)

		# set last class vector
		cvlast = cv2

	# interpolate between class vectors of bin size [smooth_factor] to smooth
	# frames
	class_vectors = smooth(class_vectors, smooth_rate(smooth_factor))

	# save record of vectors for current video
	# TODO: have deezer_id prepended to file for saving in s3
	# np.save('class_vectors.npy', class_vectors)
	# np.save('noise_vectors.npy', noise_vectors)

	return noise_vectors, class_vectors


def generate_images(video_id, noise_vectors, class_vectors, resolution,
					truncation):
	"""
	Take vectors from song_analysis and generate images

	Inputs: 
			video_id: STR; used to make unique files, avoids overwriting files
			noise_vectors: NUMPY ARRAY; formed during song analysis
			class_vectors: NUMPY ARRAY; formed during song analysis
			resolution: STR; 128, 256, 512; determines resolution of video
			truncation: FLOAT 0 to 1; should be same as passed to song analysis

	Output: 
			tmp_folder_path: points to location of frames on disk

	"""
	# convert to Tensor
	noise_vectors = torch.Tensor(np.array(noise_vectors))
	class_vectors = torch.Tensor(np.array(class_vectors))

	# initialize bigGAN model
	model = model_resolution(resolution=resolution)

	# send to CUDA if running on GPU - YOU SHOULD REALLY DO THIS
	model = model.to(device)
	noise_vectors = noise_vectors.to(device)
	class_vectors = class_vectors.to(device)

	# adds temp folder for saving frames on local disk
	tmp_folder_path = os.path.join(os.getcwd(), f"{video_id}_frames")

	if os.path.exists(tmp_folder_path):
		shutil.rmtree(tmp_folder_path)
	os.mkdir(tmp_folder_path)
	counter = 0

	for i in tqdm(range(song_duration())):

		# print progress
		pass

		if (i + 1) * batch_size > len(class_vectors):
			torch.cuda.empty_cache()
			break

		# get batch
		noise_vector = noise_vectors[i * batch_size:(i + 1) * batch_size]
		class_vector = class_vectors[i * batch_size:(i + 1) * batch_size]

		with torch.no_grad():
			output = model(noise_vector, class_vector, truncation)

		#generates image frames as numpy array
		output_cpu = output.cpu().data.numpy()

		# convert to image array and add to file containing frames
		for out in output_cpu:
			im = np.array(toimage(out))
			imsave(os.path.join(tmp_folder_path, str(counter) + ".jpg"), im)
			counter = counter + 1

		# empty cuda cache
		torch.cuda.empty_cache()

	return tmp_folder_path


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
					"ContentType": "image/jpeg"
				}
			)

	except ClientError as e:
		logging.error(e)
		return "error uploading"

	return 'Succesfully uploaded files to S3'


def save_video(tmp_folder_path, song, outname):
	"""
	Input: 
			tmp_folder_path: STR; path of folder containing images for frames
			song: STR; location of song
			outname: STR; desired name of files

	Output: 
			creates video in mp4 format, and jpg for thumbnail and calls to save

	"""
	files_path = [os.path.join(tmp_folder_path, x)
					for x in os.listdir(tmp_folder_path) if x.endswith('.jpg')]

	files_path.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	aud = mpy.AudioFileClip(song, fps=44100)
	aud.duration = 30

	# creates mp4
	clip = mpy.ImageSequenceClip(files_path, fps=22050 / frame_length)
	clip = clip.set_audio(aud)
	clip.write_videofile(outname + ".mp4", audio_codec='aac')

	# saves thumbnail
	os.rename(files_path[-1], outname + ".jpg")

	# cleans temp directory
	if os.path.exists(tmp_folder_path):
		shutil.rmtree(tmp_folder_path)

	return upload_file_to_s3(outname + ".mp4", outname + ".jpg")
