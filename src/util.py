import skimage.io
import skimage.transform
import numpy as np
import cv2
import os

def data_augmentation(file):
    image = cv2.imread(os.path.join(image_path,file))
    horizontal_img = image.copy()
    horizontal_img = cv2.flip(horizontal_img,1)
    newname = os.path.splitext(file)[0]+"_flip.jpg"
    cv2.imwrite(os.path.join(image_path,newname),horizontal_img)
    return newname

def load_image( path ):
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    img /= 255.
    # crop image
    """
    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    """
    resized_img = skimage.transform.resize(img,[224,224])
    return resized_img

