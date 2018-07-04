# make sure detection.py is in program folder
from  detection import Detector
import cv2

# given paths from both models to Detector , and It will load the model on 
# your memory (or gpu memory).
#det_mod_path = 'models/det_nets_3.ckpt'
#cal_mod_path = 'models/cal_nets_14.ckpt'

img_path = '/home/rapela/Downloads/redes/CNN_Face_Detection/rapela.jpg'
image = cv2.imread(img_path)

det_mod_path = 'models/48_net_2.ckpt' # det_nets_3
cal_mod_path = 'models/48_cal_net_2.ckpt' # 48_cal_net_3.ckpt.data-00000-of-00001
detector = Detector(det_mod_path,cal_mod_path)


bboxes = detector.img_pyramids(image)
print(len(bboxes))
bboxes = detector.predict(image,bboxes,net = 'net48',threshold = 0.9)
#print(len(bboxes))
bboxes = detector.non_max_sup(bboxes,iou_thresh = 0.1)
print(len(bboxes))

h , w = image.shape[:2]

#bboxes = detector.detect(img)
import matplotlib.pyplot as plt
c = 0
for b in bboxes:
	xmin,ymin,xmax,ymax,prop = b[:]
	print (prop)
	if prop > 0.8:
		cv2.rectangle(image, (int(xmin*w), int(ymin*h)), (int(xmax*w), int(ymax*h)), (255, 0, 0), 2)
		c = c + 1
print(c)
cv2.imshow("Window", image)
cv2.waitKey(0)
#if cv2.waitKey(0) == 27:
#	break

cv2.destroyAllWindows()