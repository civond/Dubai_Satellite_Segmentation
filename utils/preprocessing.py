import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image

root_directory = './raw_data'
patch_size = 256 # 256x256 pixels

for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':   #Find all 'images' directories
        images = os.listdir(path)  #List of all image names in this subdirectory
        for i, image_name in enumerate(images):  
            if image_name.endswith(".jpg"):   #Only read jpg images...
               
                image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                image = np.array(image)
       
                #Extract patches from each image
                print("Processing", path+"/"+image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
                
                path_temp = path.split('\\')[1]
                name_temp = image_name.split('.')[0].split('_')[-1]
                counter=0
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                    
                        single_patch_img = patches_img[i,j,:,:]
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.

                        patch_name = os.path.join("data", f"{path_temp}_{name_temp}_{counter}.jpg")
                        print("\t" + patch_name)
                        cv2.imwrite(patch_name, single_patch_img)
                        counter+=1

# prepare masks              
for path, subdirs, files in os.walk(root_directory):
    #print(path)  
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':   #Find all 'images' directories
        masks = os.listdir(path)  #List of all image names in this subdirectory
        for i, mask_name in enumerate(masks):  
            if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
                
                
                mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as Grey (or color but remember to map each color to an integer)
                #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                mask = np.array(mask)             

                #Extract patches from each image
                print("Processing mask:", path+"/"+mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

                mask_path_temp = path.split('\\')[1]
                mask_name_temp = mask_name.split('.')[0].split('_')[-1]
                counter = 0

                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        
                        single_patch_mask = patches_mask[i,j,:,:]
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                        single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.     

                        patch_name = os.path.join("data", f"{mask_path_temp}_{mask_name_temp}_{counter}_mask.png")
                        cv2.imwrite(patch_name, single_patch_mask)
                        print("\t" + patch_name)
                        counter+=1
