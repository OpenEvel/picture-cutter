#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
модуль utils
Реализует полезные функция и классы для работы с изображениями
"""
import cv2
import numpy as np
from copy import deepcopy
import os

p = 64
black = 0

def lens_segments(seq, black = 0):
    lens_white = [0]

    before_color = seq[0]

    if before_color > black:
        lens_white[-1] += 1
        i_last_white = 0
    else:
        i_last_white = -1

    for i in range(1, len(seq)):
        cur_color = seq[i]
        if cur_color > black:
            if before_color > black:
                lens_white[-1] += 1
            else:
                if i_last_white - i == 2:
                     lens_white[-1] += 2
                else:
                    lens_white.append(1)
                i_last_white = i
                
        before_color = cur_color

    return lens_white

def get_slice(i, top_seq):
    try:
        start = i.start
        stop  = i.stop
        step  = i.step
    except AttributeError:        
        return None
    else:
        if start==None:
            start = 0
        if start < 0:
            start = top_seq - abs(start)

        if stop==None:
            stop = top_seq
        if stop < 0:
            stop = top_seq - abs(stop)

        if step==None:
            step = 1

        return slice(start, stop, step)

class ImageIterator:
    def __init__(self, img):
        self.img = img
        self.start = 0
        self.stop = len(img)

    def __next__(self):
        if self.start == self.stop:
            raise StopIteration
        else:
            self.start += 1
            return self.img[self.start-1]


class Image:
    def __init__(self, img, p_y=p, p_x=p):
        self.img = deepcopy(img)
        self.y_size = self.img.shape[0]
        self.x_size = self.img.shape[1]
        self.p_y = p_y        
        self.p_x = p_x

    @classmethod
    def new(cls, y_size, x_size, depth=np.uint8, chennals=3, p_y=p, p_x=p):
        img = np.zeros((y_size, x_size, chennals), depth)
        return cls(img, p_y, p_x)

    @classmethod
    def open(cls, file, p_y=p, p_x=p):
        if not os.path.isfile(file):
            message = f'"{file}" не существует или не является файлом'
            raise ValueError(message)

        img = cv2.imread(file)
        return cls(img, p_y, p_x)

    def save(self, file):
        cv2.imwrite(file, self.img)

    @property
    def rows(self):
        return self.y_size // self.p_y
    
    @property
    def cols(self):
        return self.x_size // self.p_x    

    def __len__(self):
        return self.cols*self.rows

    def __getitem__(self, index):

        s = get_slice(index, len(self))

        if not s:
            # index - это целое число
            if index < 0:
                index = len(self) - abs(index)

            if index < 0  or index > len(self):
                message =  f"Вышли за пределы картиники [0:{len(self)}], index={index}"
                raise IndexError(message)

            row, col = divmod(index, self.cols)

            left  = col*self.p_x
            top   = row*self.p_y
            right = left + self.p_x
            down  = top + self.p_y

            block = self.img[top:down, left:right]

            cls = self.__class__

            return cls(block,  p_y=self.p_y, p_x=self.p_x)
        else:
            # index - это slice
            mas = []
            for i in range(s.start, s.stop, s.step):
                mas.append(self[i])
        return mas

    def __setitem__(self, index, block):
        block = deepcopy(block)
        if index < 0:
            index = len(self) - abs(index)

        if index < 0  or index > len(self):
            message =  f"Вышли за пределы картиники [0:{len(self)}], index={index}"
            raise IndexError(message)

        rol, col = divmod(index, self.cols)

        left  = col*self.p_x
        top   = rol*self.p_y
        right = left + self.p_x
        down  = top + self.p_y

        if block.x_size != self.p_x or block.y_size != self.p_y:
            message = 'Размеры блока %dx%d не подходят для вставки в %dx%d' % (block.y_size, block.x_size, self.p_y, self.p_x)
            raise ValueError(message)

        self.img[top:down, left:right] = block.img

    def __iter__(self):
        return ImageIterator(self)

    def __add_top(self, block):
        """
        добавить блок сверху
        """
        if block.x_size != self.x_size:
            message = f'Блоки не стыкуются по ширине my={self.x_size} other={block.x_size}'
            ValueError(message)

        cls = self.__class__

        new_img = cls.new(y_size = self.y_size + block.y_size, x_size = self.x_size)
        new_img.p_x = block.x_size & self.x_size

        top = cls(block.img, p_x=new_img.p_x)
        down = cls(self.img, p_x=new_img.p_x)

        cnt_parts = 0
        for i in range(len(top)):
            new_img[cnt_parts] = top[i]
            cnt_parts += 1
        for i in range(len(down)):
            new_img[cnt_parts] = down[i]
            cnt_parts += 1

        new_img.p_x = min(self.p_x, block.p_x)
        return new_img

    def __add_down(self, block):
        """
        добавить блок снизу
        """
        if block.x_size != self.x_size:
            message = f'Блоки не стыкуются по ширине my={self.x_size} other={block.x_size}'
            ValueError(message)

        cls = self.__class__

        new_img = cls.new(y_size = self.y_size + block.y_size, x_size = self.x_size)
        new_img.p_x = block.x_size & self.x_size

        top = cls(self.img, p_x=new_img.p_x)
        down = cls(block.img, p_x=new_img.p_x)
        cnt_parts = 0
        for i in range(len(top)):
            new_img[cnt_parts] = top[i]
            cnt_parts += 1
        for i in range(len(down)):
            new_img[cnt_parts] = down[i]
            cnt_parts += 1

        new_img.p_x = min(self.p_x, block.p_x)
        return new_img

    def __add_left(self, block):
        """
        добавить блок слева
        """
        if block.y_size != self.y_size:
            message = f'Блоки не стыкуются по высоте my={self.y_size} other={block.y_size}'
            ValueError(message)

        cls = self.__class__

        new_img = cls.new(y_size = self.y_size, x_size = self.x_size + block.x_size)
        new_img.p_y = block.y_size & self.y_size

        left = cls(block.img, p_y=new_img.p_y)
        right = cls(self.img, p_y=new_img.p_y)

        cnt_parts = 0
        for i in range(len(left)):
            new_img[cnt_parts] = left[i]
            cnt_parts += 1
        for i in range(len(right)):
            new_img[cnt_parts] = right[i]
            cnt_parts += 1

        new_img.p_y = min(self.p_y, block.p_y)
        return new_img

    def __add_right(self, block):
        """
        добавить блок спрва
        """
        if block.y_size != self.y_size:
            message = f'Блоки не стыкуются по высоте my={self.y_size} other={block.y_size}'
            ValueError(message)

        cls = self.__class__

        new_img = cls.new(y_size = self.y_size, x_size = self.x_size + block.x_size)
        new_img.p_y = block.y_size & self.y_size

        left = cls(self.img, p_y=new_img.p_y)
        right = cls(block.img, p_y=new_img.p_y)

        cnt_parts = 0
        for i in range(len(left)):
            new_img[cnt_parts] = left[i]
            cnt_parts += 1
        for i in range(len(right)):
            new_img[cnt_parts] = right[i]
            cnt_parts += 1

        new_img.p_y = min(self.p_y, block.p_y)
        return new_img

    def add(self, block, mode):
        mode = mode[0].lower()
        if mode == 't': return self.__add_top(block)
        elif mode == 'd': return self.__add_down(block)
        elif mode == 'l': return self.__add_left(block)
        elif mode == 'r': return self.__add_right(block)
        else:
            message = f"Параметр mode='{mode}' не в множестве ['t', 'd', 'l', 'r']"
            raise ValueError(message)

    def get_contours(self, thr1=0, thr2=50):
        if len(self.img.shape) > 2:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.img
        return cv2.Canny(gray, thr1, thr2)

    def get_threshold(self):
        if len(self.img.shape) > 2:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.img

        return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)


    def is_my(self, block, mode, thr1=30, thr2=50):
        mode = mode[0].lower()

        if mode in ('l', 't'):
            block = block[-1]
            new_img = self[0]
        else:
            block = block[0]
            new_img = self[-1]

        new_img = new_img.add(block, mode)

        # blur = cv2.GaussianBlur(new_img.img, (5,5), 0)
        contours = new_img.get_contours(thr1, thr2)
        # th = new_img.get_threshold()

        if mode in ('l' , 'r'):
            line_pixels = contours[:, -2 + new_img.x_size//2: new_img.x_size//2 + 2]
            line_len = new_img.y_size
        else:
            line_pixels = contours[-2 + new_img.y_size//2: new_img.y_size//2 + 2, :]
            line_len = new_img.x_size

        cuts = []        
        for i in range(2):
            if mode in ('l', 'r'):
                mas_cuts = [line_pixels[k, i] | line_pixels[k, i+1] for k in range(line_len)]
            else:
                mas_cuts = [line_pixels[i, k] | line_pixels[i+1, k] for k in range(line_len)]
            cuts.append(max(lens_segments(mas_cuts)))

        # cv2.imshow("Contours", contours)
        # cv2.imshow("Treshold", th)
        # cv2.waitKey(0)

        prob_simple = self.is_my_simple(block, mode)
        my_prob = 1 - max(cuts)/line_len

        if prob_simple <= 0.65:
            return prob_simple*my_prob

        return my_prob

    def is_my_simple(self, block, mode):
        mode = mode[0].lower()

        if mode in ('l', 't'):
            block = block[-1]
            new_img = self[0]
        else:
            block = block[0]
            new_img = self[-1]

        new_img = new_img.add(block, mode)

        if mode in ('l' , 'r'):
            lol = new_img.img[:, -1 + new_img.x_size//2: new_img.x_size//2 + 1]
            line_len = new_img.y_size
        else:
            lol = new_img.img[-1 + new_img.y_size//2: new_img.y_size//2 + 1, :]
            line_len = new_img.x_size

        a = []
        for i in range(line_len):
            temp = []
            for k in range(3):
                if mode in ('l', 'r'):
                    temp.append(abs(lol[i,0][k]-lol[i,1][k]))
                else:
                    temp.append(abs(lol[0,i][k]-lol[1,i][k]))
            a.append(sum(temp))
        # print(a)
        cut = max(lens_segments(a, 7))

        return (1 - cut/line_len)


    def color(self):
        """
        возвращает доминирующий цвет
        """
        Z = self.img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 1
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((self.img.shape))
        return res2[0,0]

    def sort_color(self):
        """
        сортировка по цвету
        самые светлые в начале
        """
        mas = [self[i] for i in range(len(self))]
        mas.sort(key=(lambda img: tuple(img.color())), reverse=True)
        return mas

    def is_equal(self, block):
        gray1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(block.img, cv2.COLOR_BGR2GRAY)

        rows = gray1.shape[0]
        cols = gray2.shape[1]
        for i in range(rows):
            for j in range(cols):
                if gray1[i,j] != gray2[i,j]:
                    return False
        return True

    def show(self, window_name='Image'):
        cv2.imshow(str(window_name), self.img)
        cv2.waitKey(0)

    def pop(self, index):

        if index < 0:
            index = len(self) - abs(index)

        new_img = None
        for i in range(len(self)):
            if i != index:
                cur_img = self[i]
                if not new_img:
                    new_img = cur_img
                else:
                    new_img = new_img.add(cur_img, 'r')

        self.__init__(new_img.img, p_y = self.p_y, p_x=self.p_x)

class MImage(Image):
    """
    Класс чтобы обращаться к элементу изображения как к элементу матрицы
    то есть обращаемся к элементу через 2 индекса
    -------------------------------------------------
    старое обращение к элементу через один индекс сохраняется
    """

    def __getitem__(self, indexs):

        try:
            indexs = tuple(indexs)
        except TypeError:
            return super().__getitem__(indexs)

        cnt_indexs = len(indexs)
        if cnt_indexs==2:

            row, col = indexs

            s_row = get_slice(row, self.rows)
            if (not s_row) and row < 0:
                row = self.rows - abs(row)

            s_col = get_slice(col, self.cols)
            if (not s_col) and col < 0:
                col = self.cols - abs(col)

            cls = self.__class__

            if not s_row and not s_col:

                left  = col*self.p_x
                top   = row*self.p_y
                right = left + self.p_x
                down  = top + self.p_y

                block = self.img[top:down, left:right]
                return cls(block,  p_y=self.p_y, p_x=self.p_x)

            elif not s_col:
                img = None
                for i in range(s_row.start, s_row.stop, s_row.step):
                    cur_img = self[i, col]
                    if not img:
                        img = cur_img
                    else:
                        img = img.add(cur_img, 'd')
                return img

            elif not s_row:
                img = None
                for j in range(s_col.start, s_col.stop, s_col.step):
                    cur_img = self[row, j]
                    if not img:
                        img = cur_img
                    else:
                        img = img.add(cur_img, 'r')
                return img
            else:
                row_img = None
                for i in range(s_row.start, s_row.stop, s_row.step):

                    col_img = self[i, s_col.start:s_col.stop:s_col.step]
                    if not row_img:
                        row_img = col_img
                    else:
                        row_img = row_img.add(col_img, 'd')

                return row_img

        else:
            raise IndexError(f'bed index {indexs}')

    def __setitem__(self, indexs, block):

        block = deepcopy(block)
        try:
            indexs = tuple(indexs)
        except TypeError:
            super().__setitem__(indexs, block)
        else:
            cnt_indexs = len(indexs)
            if cnt_indexs==2:

                row, col = indexs

                s_row = get_slice(row, self.rows)
                if (not s_row) and row < 0:
                    row = self.rows - abs(row)

                s_col = get_slice(col, self.cols)
                if (not s_col) and col < 0:
                    col = self.cols - abs(col)

                cls = self.__class__

                if not s_row and not s_col:
                    left  = col*self.p_x
                    top   = row*self.p_y
                    right = left + self.p_x
                    down  = top + self.p_y                
                    self.img[top:down, left:right] = block.img

                elif not s_col:
                    block.p_x = block.x_size
                    temp_p_x = self.p_x
                    self.p_x = block.p_x
                    block.p_y = self.p_y

                    i_bl = 0
                    for i in range(s_row.start, s_row.stop, s_row.step):
                        self[i, col] = block[i_bl, 0]
                        i_bl+=1

                    self.p_x = temp_p_x

                elif not s_row:
                    block.p_y = block.y_size
                    temp_p_y = self.p_y
                    self.p_y = block.p_y
                    block.p_x = self.p_x

                    j_bl = 0
                    for j in range(s_col.start, s_col.stop, s_col.step):
                        self[row, j] = block[0, j_bl]
                        j_bl += 1
                    self.p_y = temp_p_y
                else:
                    block.p_x = self.p_x
                    block.p_y = self.p_y
                    i_bl = 0
                    for i in range(s_row.start, s_row.stop, s_row.step):
                        j_bl = 0
                        for j in range(s_col.start, s_col.stop, s_col.step):
                            self[i,j] = block[i_bl, j_bl]
                            j_bl += 1
                        i_bl += 1
            else:
                raise IndexError(f'bed index {indexs}')


def find_best_combination(img, mas, mode='r'):
    if not mas: return None, None, -1

    i_forward = 0
    max_prob_forward = img.is_my(mas[i_forward], mode)

    i_back = 0
    max_prob_back = mas[i_back].is_my(img, mode)

    for i in range(1, len(mas)):

        cur_prob_forward = img.is_my(mas[i], mode)
        if cur_prob_forward > max_prob_forward:
            max_prob_forward = cur_prob_forward
            i_forward = i

        cur_prob_back = mas[i].is_my(img, mode)
        if cur_prob_back > max_prob_back:
            max_prob_back = cur_prob_back
            i_back = i

    if max_prob_forward > max_prob_back:
        res_img = img.add(mas[i_forward], mode)
        index = i_forward
        prob = max_prob_forward
    else:
        res_img = mas[i_back].add(img, mode)
        index = i_back
        prob = max_prob_back

    return (res_img, index, prob)

def find_parts(image, *, mode='r', parts2 = [], parts3 = []):
    parts1 = [image[i] for i in range(len(image))]

    # пока у нас есть не скреплённые части изображения

    while(parts1):
        cur_img = parts1.pop(0)

        img2, i1, prob1 = find_best_combination(cur_img, parts1, mode)
        img3, i2, prob2 = find_best_combination(cur_img, parts2, mode)

        img4, i3, prob3 = find_best_combination(cur_img, parts3, mode)

        p = max(prob1, prob2, prob3)
        if p == prob1:
            parts1.pop(i1)
            parts2.append(img2)
        elif p == prob2:
            parts2.pop(i2)
            parts3.append(img3)
        else:
            parts3.pop(i3)
            my_mode = 'r' if mode in ('r', 'l') else 'd'
            new1 = img4[:2]
            new2 = img4[-2:]

            new1 = new1[0].add(new1[1], my_mode)
            new2 = new2[0].add(new2[1], my_mode)

            parts2.append(new1)
            parts2.append(new2)

    return parts2, parts3

def one_to_two(iterable, *, parts2 = [], mode='r'):
    lol = iter(iterable)
    try:
        while True:
            one = next(lol)
            two = next(lol)

            x = one.add(two, mode)
            parts2.append(x)
    except StopIteration:
        pass

    return parts2

def find_2(*, parts2=[], parts3=[], mode='r'):
    while(parts3):
        cur_img = parts3.pop(0)
        img5, i3, prob3 = find_best_combination(cur_img, parts3, mode)
        parts3.pop(i3)
        img8, i2, prob8 = find_best_combination(img5, parts2, mode)
        parts2.pop(i2)

        lol = [img8[i] for i in range(len(img8))]
        parts2 = one_to_two(lol, parts2=parts2, mode=mode)

    return parts2
