import utils as u
from copy import deepcopy
import cv2

file_image = 'temp.png'
res_file_image = 'lol.png'

before_p_x = u.p
before_p_y = u.p

mode = 'd'
after_p_x = u.p
after_p_y = u.p*2


if __name__ == '__main__':

    parts1 = []
    parts2 = []
    parts3 = []

    before = u.MImage.open(file_image)
    
    before.p_x = before_p_x
    before.p_y = before_p_y

    # массив клеток еденичной длины
    # parts1.append(before[4,2])
    # parts1.append(before[5,2])
    # parts1.append(before[6,1])
    # parts1.append(before[7,1])
    
    # for img in parts1:
    #     img.show('1')
    # cv2.destroyAllWindows()


    # массив клеток удвоенных
    # parts2.append(before[4,3:5])

    # for img in parts2:
    #     img.show('2')
    # cv2.destroyAllWindows()

    # массив клеток утроенных
    # parts3.append(copy(before[6, 5:8]))
    # parts3.append(copy(before[6, 2:5]))

    # for img in parts3:
    #     img.show('3')
    # cv2.destroyAllWindows()

    # собираем оставшиеся клетки на картинке
    # в массив двоек
    before.p_x = u.p*8   
    lines=[before[i] for i in range(len(before))]

    # for line in lines:
    #     line.show('line before')
    # cv2.destroyAllWindows()

    before.p_x = before_p_x
    before.p_y = before_p_y

    for line in lines:
        line.p_x = before_p_x
        line.p_y = before_p_y

    # lines[4].pop(2)
    # lines[5].pop(2)
    # lines[6].pop(1)
    # lines[7].pop(1)

    for line in lines:
        line.show('line after')

    for i in range(0,len(lines), 2):
        for j in range(len(lines[i])):
            parts2.append(lines[i][j].add(lines[i+1][j], mode))
        # parts2 = u.one_to_two(lol, parts2=parts2, mode='d')
    cv2.destroyAllWindows()

    # for img in parts2:
    #     img.show('2')
    # cv2.destroyAllWindows()

    r2, r3 = u.find_parts(deepcopy(parts1), mode=mode, parts2=deepcopy(parts2), parts3=deepcopy(parts3))

    # for img in r2:
    #     img.show(2)
    # cv2.destroyAllWindows()
    # for img in r3:
    #     img.show(3)
    # cv2.destroyAllWindows()

    # before.p_x=p_x_one
    # before.p_y=u.p*2
    # for i in range(len(before)):
    #     before[i] = r2[i]

    before.show()
    before.save(res_file_image)