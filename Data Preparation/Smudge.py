# Note: Image name will be stored as "Smudge_OriginalName" to avoid confict
import cv2
import numpy as np
import glob

def basicTransform(img):
	_, mask = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
	img = cv2.bitwise_not(mask)
	return img

PATH_TO_DEST = "/home/elimen/Data/cascade-tabnet_pytorch/Data Preparation/test_results/"
PATH_TO_ORIGIAL_IMAGES = "/home/elimen/Data/cascade-tabnet_pytorch/Data Preparation/test_images/"

img_files = glob.glob(PATH_TO_ORIGIAL_IMAGES+"*.*")

total = len(img_files)
for count,i in enumerate(img_files):
  image_name = i.split("/")[-1]
  print("Progress : ",count,"/",total)
  img = cv2.imread(i)
  
  # Split the 3 channels into Blue,Green and Red
  b,g,r = cv2.split(img)
  
  # Apply Basic Transformation
  b = basicTransform(b)
  r = basicTransform(r)
  g = basicTransform(g)
  cv2.imwrite(PATH_TO_DEST+"/basicTrans_b_"+image_name,b)
  cv2.imwrite(PATH_TO_DEST+"/basicTrans_r_"+image_name,r)
  cv2.imwrite(PATH_TO_DEST+"/basicTrans_g_"+image_name,g)
  
  # Perform the distance transform algorithm
  b = cv2.distanceTransform(b, cv2.DIST_L2, 5)  # ELCUDIAN
  g = cv2.distanceTransform(g, cv2.DIST_L1, 5)  # LINEAR
  r = cv2.distanceTransform(r, cv2.DIST_C, 5)   # MAX
  cv2.imwrite(PATH_TO_DEST+"/distanceTrans_b_"+image_name,b)
  cv2.imwrite(PATH_TO_DEST+"/distanceTrans_r_"+image_name,r)
  cv2.imwrite(PATH_TO_DEST+"/distanceTrans_g_"+image_name,g)

  # Normalize
  r = cv2.normalize(r, r, 0, 1.0, cv2.NORM_MINMAX)
  g = cv2.normalize(g, g, 0, 1.0, cv2.NORM_MINMAX)
  b = cv2.normalize(b, b, 0, 1.0, cv2.NORM_MINMAX)
  cv2.imwrite(PATH_TO_DEST+"/normalize_b_"+image_name,b)
  cv2.imwrite(PATH_TO_DEST+"/normalize_r_"+image_name,r)
  cv2.imwrite(PATH_TO_DEST+"/normalize_g_"+image_name,g)

  # Merge the channels
  dist = cv2.merge((b,g,r))
  dist = cv2.normalize(dist,dist, 0, 4.0, cv2.NORM_MINMAX)
  dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)

  # In order to save as jpg, or png, we need to handle the Data
  # format of image
  data = dist.astype(np.float64) / 4.0
  data = 1800 * data # Now scale by 1800
  dist = data.astype(np.uint16)

  # Save to destination
  cv2.imwrite(PATH_TO_DEST+"/Smudge_"+image_name,dist)
