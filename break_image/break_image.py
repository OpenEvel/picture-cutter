#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# добавлем родительский каталог в пути посика модуле
# чтобы был виден модуль utils
import os
import sys
sys.path = [os.pardir] + sys.path

# чтобы можно было загрузить изображение по частям
from utils import Image

# чтобы перемешать части изображения
import random


orig = Image.open('cat.png')

img = Image.new(orig.x_size, orig.y_size)

parts = [orig[i] for i in range(len(orig))]
# parts = [image for image in orig]

random.shuffle(parts)

for i in range(len(orig)):
    img[i] = parts[i]

img.save('cat_break.png')
img.show('cat_break')