from copy import deepcopy
import utils as u
import cv2

file_image = 'temp.png'
# res_file_image = ''

before_p_x = u.p
before_p_y = u.p

mode = 'r'
after_p_x = u.p*2
after_p_y = u.p


img = u.MImage.open(file_image)
img.p_x = before_p_x
img.p_y = before_p_y

img[1:3, :2] = img[1:3, -2:]

img.show()
# img.save('lol.png')