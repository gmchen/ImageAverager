import numpy as np
import cv2
from os import listdir

# Path to image directory
image_dir = "/home/greg/Dropbox/Greg/Python/ImageAverager/faces/All/"
filenames = listdir(image_dir)
filenames.sort()

imgs = []
for f in filenames:
	imgs.append(cv2.imread(image_dir + f,0))

max_height = 0
max_width = 0
for i in imgs:
	if i.shape[0] > max_height:
		max_height = i.shape[0]
	if i.shape[1] > max_width:
		max_width = i.shape[1]

centered_imgs = []
for i in imgs:
	height, width = i.shape
	new_img = np.zeros([max_height, max_width], np.uint8)
	new_img.fill(255)	# white background
	start_height = (max_height - height) / 2
	start_width = (max_width - width) / 2
	new_img[start_height:(start_height + height), start_width:(start_width + width)] = i
	centered_imgs.append(new_img)

final_img = np.zeros([max_height, max_width], np.uint8)
for i in range(len(centered_imgs)):
	final_img = cv2.addWeighted(final_img, i/(i+1.0), centered_imgs[i], 1/(i+1.0), 0)

cv2.imshow('Final Image', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
