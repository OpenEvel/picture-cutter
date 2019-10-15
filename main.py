"""
Главная программа - пытаемся восстановить изображение
"""
import utils as u
import cv2

file_image = 'image.png'
res_file_image = 'temp.png'

before_p_x = u.p
before_p_y = u.p

mode = 'r'
after_p_x = u.p*2
after_p_y = u.p

if __name__ == '__main__':

    before = u.MImage.open(file_image)
    before.p_y = before_p_y
    before.p_x = before_p_x

    before.show('before')
    cv2.destroyAllWindows()

    mas_before = before.sort_color()
    r2, r3 = u.find_parts(mas_before, mode=mode)
    # r2, r3 = u.find_parts(before, mode=mode)
    print('r3 = ', len(r3))
    print('r2 = ', len(r2))
    print('r3*3 + r2*2 = ', len(r3)*3+len(r2)*2)

    # for img in r2:
    #     img.show('2')
    # cv2.destroyAllWindows()

    # for img in r3:
    #     img.show('3')
    # cv2.destroyAllWindows()

    r2 = u.find_2(parts2=r2, parts3=r3, mode=mode)

    after = u.Image.new(before.y_size, before.x_size)
    after.p_x = after_p_x
    after.p_y = after_p_y

    # for img in r2:
    #     img.show('2')
    # cv2.destroyAllWindows()

    for i in range(len(after)):
        after[i] = r2[i]

    after.show('after')
    after.save(res_file_image)