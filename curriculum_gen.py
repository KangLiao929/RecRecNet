from tempfile import TemporaryFile
from PIL import Image
import numpy as np
import glob
import os.path
import random as rd
import cv2
import argparse


def load_random_image(path_source, size):
    img_path = rd.choice(glob.glob(os.path.join(path_source, '*.jpg'))) 
    img = Image.open(img_path)   
    img = img.resize(size)             
    img_data = np.asarray(img)
    return img_data


def save_to_file(path_dest, index, img, warped_img):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    img_path = path_dest + '/gt/' 
    warped_img_path = path_dest + '/input/'
    
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(warped_img_path):
        os.makedirs(warped_img_path)

    gt_path = img_path + index + '.jpg'
    input_path = warped_img_path+ index + '.jpg'
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    warped_img = Image.fromarray(warped_img.astype('uint8')).convert('RGB')
    img.save(gt_path)
    warped_img.save(input_path)


def generate_dataset(path_source, path_dest, data, dof):

    if(dof==4):
        for count in range(0, data):
            img = load_random_image(path_source, [256, 256]).astype(np.int16)
            offset_w = rd.randint(230, 245)
            offset_h = rd.randint(230, 245)
            src = cv2.resize(img, (offset_h, offset_w))

            init_x = int((256 - offset_w) / 3) 
            init_y = int((256 - offset_h) / 3)
            init_x += rd.randint(-init_x, init_x)
            init_y += rd.randint(-init_y, init_y)

            img_warped = np.zeros((256, 256, 3))
            img_warped.fill(255)
            img_warped[init_x : init_x + offset_w, init_y : init_y + offset_h] = src
            
            save_to_file(path_dest, str(count+1).zfill(6), img, img_warped)

    if(dof==8):
        for count in range(0, data):
            img = load_random_image(path_source, [256, 256]).astype(np.int16)

            src_input1 = np.zeros([4, 2])
            dst = np.zeros([4, 2])

            #Upper left
            src_input1[0][0] = 0
            src_input1[0][1] = 0
            # Upper right
            src_input1[1][0] = 256
            src_input1[1][1] = 0
            # Lower left
            src_input1[2][0] = 0
            src_input1[2][1] = 256
            # Lower right
            src_input1[3][0] = 256
            src_input1[3][1] = 256

            offset = np.empty(8, dtype=np.int8)
            #The position of each vertex after the coordinate perturbation

            for j in range(8):
                offset[j] = rd.randint(0, 20)
            
            # Upper left
            alpha = np.random.randint(0,2)
            dst[0][0] = src_input1[0][0] + offset[0]*alpha
            dst[0][1] = src_input1[0][1] + offset[1]*(1-alpha)
            # Upper righ
            dst[1][0] = src_input1[1][0] - offset[2]*(1-alpha)
            dst[1][1] = src_input1[1][1] + offset[3]*alpha
            # Lower left
            dst[2][0] = src_input1[2][0] + offset[4]*(1-alpha)
            dst[2][1] = src_input1[2][1] - offset[5]*alpha
            # Lower right
            dst[3][0] = src_input1[3][0] - offset[6]*alpha
            dst[3][1] = src_input1[3][1] - offset[7]*(1-alpha)


            h, _ = cv2.findHomography(src_input1, dst)
            img_warped = np.asarray(cv2.warpPerspective(img-255, h, (256, 256))).astype(np.uint8)
            img_warped = img_warped + 255
            mask_warped = np.asarray(cv2.warpPerspective(np.ones_like(img), h, (256, 256))).astype(np.uint8)
            img_warped = (1-mask_warped)*255 + img_warped
            
            save_to_file(path_dest, str(count+1).zfill(6), img, img_warped)

    else:
        raise NotImplementedError('You must choose the dof from 2, 4, or 8.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, default='imgs/', help='path of the source image dataset')
    parser.add_argument('--path2', type=str, default='imgs/', help='path of the synthesized curriculum dataset')
    parser.add_argument('--size', type=int, default=5000, help='size of curriculum dataset')
    parser.add_argument('--dof', type=int, default=4, help='dof of the curriculum dataset')

    ''' parser configs '''
    args = parser.parse_args()
    
    print("Generating dataset...")

    generate_dataset(args.path1, args.path2, args.size, args.dof)
