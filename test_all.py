import argparse
import numpy as np
import tensorflow as tf
import os
import cv2
import glob
import time
from inpaint_model import WNet
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from config import Config

import torchvision.transforms.functional as F
import lpips

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./examples/test.jpg', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='./examples/mask.png', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='./examples/output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='./logs/Paris', type=str,
                    help='The directory of tensorflow checkpoint.')


def data_batch(list1, list2):
    test_image = tf.data.Dataset.from_tensor_slices(list1)
    test_mask = tf.data.Dataset.from_tensor_slices(list2)

    def image_fn(img_path):
        x = tf.read_file(img_path)
        x_decode = tf.image.decode_jpeg(x, channels=3)
        img = tf.image.resize_images(x_decode, [256, 256])
        img = tf.cast(img, tf.float32)
        return img

    def mask_fn(mask_path):
        x = tf.read_file(mask_path)
        x_decode = tf.image.decode_jpeg(x, channels=1)
        mask = tf.image.resize_images(x_decode, [256, 256])
        mask = tf.cast(mask, tf.float32)
        return mask

    test_image = test_image. \
        repeat(1). \
        map(image_fn). \
        apply(tf.contrib.data.batch_and_drop_remainder(1)). \
        prefetch(1)

    test_mask = test_mask. \
        repeat(1). \
        map(mask_fn). \
        apply(tf.contrib.data.batch_and_drop_remainder(1)). \
        prefetch(1)

    test_image = test_image.make_one_shot_iterator().get_next()
    test_mask = test_mask.make_one_shot_iterator().get_next()
    return test_image, test_mask


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    args = parser.parse_args()

    config_path = os.path.join('config.yml')
    config = Config(config_path)
    model = WNet(config)

    path_img = '...'
    path_mask = '...'
    path_save = '...'
    lists = sorted(os.listdir(path_img))
    list_img = list(glob.glob(path_img + '/*.jpg')) + list(glob.glob(path_img + '/*.png'))
    list_mask = list(glob.glob(path_mask + '/*.jpg')) + list(glob.glob(path_mask + '/*.png'))
    list_img.sort()
    list_mask.sort()
    list_img = list_img
    list_mask = list_mask
    # random.shuffle(list_mask)
    image, mask = data_batch(list_img, list_mask)

    image /= 255
    mask /= 255
    images_masked = (image * (1 - mask)) + mask
    
    # input of the model
    inputs = tf.concat([images_masked, mask], axis=3)

    # process outputs
    output = model.wnet_generator(inputs, 64, 8, mask)
    outputs_merged = (output * mask) + (image * (1 - mask))
    output *= 255
    outputs_merged *= 255
    image *= 255
    images_masked *= 255
    mask *= 255

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        sess.run(assign_ops)

        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        clk = 0
        loss_fn_alex = lpips.LPIPS(net='alex')
        
        for num in range(0, len(list_img)):
            gts, maskeds, outputs, mergeds, masks = sess.run([image, images_masked, output, outputs_merged, mask])
            
            gt = gts[0][:, :, ::-1].astype(np.uint8)
            masked = maskeds[0][:, :, ::-1].astype(np.uint8)
            merged = mergeds[0][:, :, ::-1].astype(np.uint8)
            out = outputs[0][:, :, ::-1].astype(np.uint8)
            
            img_psnr = psnr(merged, gt)
            img_ssim = ssim(merged, gt, multichannel=True, win_size=51)
            
            img_a = F.to_tensor(gt)*2-1.
            img_b = F.to_tensor(merged)*2-1.
            img_a = img_a.unsqueeze(0)
            img_b = img_b.unsqueeze(0)
            img_lpips = loss_fn_alex(img_a, img_b)

            avg_psnr += img_psnr
            avg_ssim += img_ssim
            avg_lpips += img_lpips.item()

            s = str(lists[num][:-4])
            cv2.imwrite(path_save + s + '.png', merged)

        avg_psnr /= len(list_img)
        avg_ssim /= len(list_img)
        avg_lpips /= len(list_img)
        print(avg_psnr, avg_ssim, avg_lpips)
        print('end!')
