
import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import random
def get_contour(bin_img):
    # get contour
    contour_img = np.zeros(shape=(bin_img.shape),dtype=np.uint8)
    contour_img += 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(bin_img[i][j]==0):
                contour_img[i][j] = 0
                sum = 0
                sum += bin_img[i - 1][j + 1]
                sum += bin_img[i][j + 1]
                sum += bin_img[i + 1][j + 1]
                sum += bin_img[i - 1][j]
                sum += bin_img[i + 1][j]
                sum += bin_img[i - 1][j - 1]
                sum += bin_img[i][j - 1]
                sum += bin_img[i + 1][j - 1]
                if sum ==  0:
                    contour_img[i][j] = 255

    return contour_img
def lower_thresholding(patch_data):
        mask = patch_data.copy()
        height,width = mask.shape
        sorted_pixel = np.sort(patch_data.ravel())
        hard_threshold = sorted_pixel[round(height*width*0.5)]
        if hard_threshold == 0:
            hard_threshold = 1
        mask[np.where(mask <= hard_threshold)] = 1
        mask[np.where(mask > hard_threshold)] = 0
        return mask
def get_car_in_the_image(patch_data):
    mask = upper_thresholding(patch_data)
    mask = counting_filter(mask)
    mask = hole_filling_and_preserve_the_max_connected_region(mask)
    return mask

def get_shadow_in_the_image(patch_data):
    src_patch_data = patch_data[40:96, 40:80]
    mask = lower_thresholding(src_patch_data)
    mask = counting_filter(mask)
    mask = hole_filling_and_preserve_the_max_connected_region(mask)
    black_map = np.zeros((128, 128))
    black_map[40:96, 40:80] = mask
    return black_map

def get_final_mask_in_the_image(patch_data):
    mask_of_car = upper_thresholding(patch_data)
    mask_of_car = counting_filter(mask_of_car)
    mask_of_shadow = lower_thresholding(patch_data)
    mask_of_shadow = counting_filter(mask_of_shadow)

    mask = mask_of_car + mask_of_shadow
    mask[np.where(mask>1)] = 1

    mask = hole_filling_and_preserve_the_max_connected_region(mask)
    return mask
def upper_thresholding(patch_data, thresh=None):
        mask = patch_data.copy()
        height,width = patch_data.shape        
        sorted_pixel = np.sort(patch_data.ravel())
        hard_threshold = sorted_pixel[round(height * width * 0.70)]
        mask[np.where(mask <= hard_threshold)] = 0
        mask[np.where(mask > hard_threshold)] = 1
        return mask
def counting_filter(image):
    # prepare parameters
    kernal_size = 7
    filter_conv = np.sum
    height, width = image.shape
    mask_counting = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            # prepare the kernal and pay attention to the edge
            b_i = max(0,i-kernal_size//2)
            e_i = min(height,i+kernal_size//2)
            b_j = max(0,j-kernal_size//2)
            e_j = min(width,j+kernal_size//2)

            image_of_the_filter_window = image[b_i:e_i,b_j:e_j]
            mask_counting[i][j] = filter_conv(image_of_the_filter_window==1)
    image[np.where(mask_counting < 15)] = 0
    return image

def hole_filling_and_preserve_the_max_connected_region(mask):
    contours, hierarchy, = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    max_contour_area = 0
    max_i = 0
    for i in range(len_contour):
        contour_area = cv2.contourArea(contours[i])
        if contour_area > max_contour_area:
            max_contour_area = contour_area
            max_i = i
    drawing = np.zeros_like(mask, np.uint8)  # create a black image
    img_contour = cv2.drawContours(drawing, contours, max_i, (1, 1, 1), -1)
    return img_contour

def KP_visual(type,img_path,b,noise_path,save_path):
    # img_HX_path = img_path.replace('imgHB', 'HB')[:-4]+'+'+type+'_last_model_deltaX.jpg'
    img_HX_path = img_path.split('/')[-1].replace('imgHB', 'HB')[:-4]+'+'+type+'_last_model_deltaX.npy'
    # HX_path = img_HX_path.replace('.jpg', '.npy')
    HX_path = os.path.join(noise_path,img_HX_path)
    save_path = save_path
    ## RGB
    Target_KP_color = [71,99,255]               
    Shadow_KP_color = [58,238,179]             
    BackGround_KP_color = [250,206,135]         
    Target_Not_KP_color = [211,211,211]         
    BackGround_Not_KP_color = [255,255,255]     
    Shadow_Not_KP_color = [238,238,224]             
    bord_target = [62,124,255]
    bord_shadow = [220,220,220]
    back = [220,220,220]
    HX_to_img = np.load(HX_path)[0]
    img = cv2.imread(img_path,0)
    img = cv2.resize(img,(128,128))
    # img_HX = cv2.imread(img_HX_path,0)
    car_mask = get_car_in_the_image(img)*255
    car_axis = np.argwhere(car_mask==255).tolist()
    car_background = np.argwhere(car_mask==0).tolist()
    shadow_mask = get_shadow_in_the_image(img)*255
    shadow_axis = np.argwhere(shadow_mask==255).tolist()
    shadow_background = np.argwhere(shadow_mask==0).tolist()
    final_mask = (shadow_mask+car_mask)
    # final_mask = car_mask+shadow_mask
    all_target_axis = np.argwhere(final_mask==255).tolist()
    all_background_axis = np.argwhere(final_mask==0).tolist()
    
    all_hx_coord = []           ## low entropy sigma in Shadow and Target Area
    tar_hx_coord = []           ## low entropy sigma in target area
    shadow_hx_coord = []        ## low entropy sigma in shadow area

    for i,v  in np.ndenumerate(HX_to_img):
        ## Find all the sigma in target area
        ## HX_to_img：16*16 Hi
        ## i:(x，y) v：value
        count = 0
        row_min = i[0]*8
        row_max = (i[0]+1)*8-1
        col_min = i[1]*8
        col_max = (i[1]+1)*8-1
        col = [col_min,col_max]
        row = [row_min,row_max]
        for k in range(row_min,row_max+1):
            for v in range(col_min,col_max+1):
                if [k,v] not in all_background_axis:
                    if i not in all_hx_coord:
                        all_hx_coord.append(i)
                    if [k,v] in shadow_axis:
                        if i not in shadow_hx_coord:
                            shadow_hx_coord.append(i)
                    elif [k,v] in car_axis:
                        if i not in tar_hx_coord:
                            tar_hx_coord.append(i)
                    else:
                        raise IndexError('Unknow KP')
    for i in tar_hx_coord:
        if i in shadow_hx_coord:
            shadow_hx_coord.remove(i)
    BG_HX = []          ## sigma in background
    TAR_HX = []         ## sigma in target
    SHADOW_HX = []      ## sigma in shadow
    for i,v  in np.ndenumerate(HX_to_img):
        if i not in all_hx_coord:
            BG_HX.append(v)
        else:
            if i in tar_hx_coord:
                TAR_HX.append([i,v])
            elif i in shadow_hx_coord:
                SHADOW_HX.append([i,v])
            else:
                raise KeyError('UnKnow sigma点')
    BG_HX_mean = np.mean(BG_HX)                ## BackGround_Mean_HX
    Shadow_number_kp = []
    Target_number_kp = []
    Target_b = []
    Shadow_b = []
    for i in TAR_HX:
        Target_number_kp.append(BG_HX_mean-i[1])
    for i in SHADOW_HX:
        Shadow_number_kp.append(BG_HX_mean-i[1])
    Target_number_kp = np.array(Target_number_kp)
    Shadow_number_kp = np.array(Shadow_number_kp)

    TAR_HX_mean = np.mean(TAR_HX)              ## Target_Mean_HX
    Shadow_HX_mean = np.mean(SHADOW_HX)
    Visual_KP = np.full((128,128,3),255)     
    All_KP = []
    All_KP_with_value = []
    for index,Hi in np.ndenumerate(HX_to_img):
        # if BG_HX_mean - Hi > b:
            All_KP.append(index)
            All_KP_with_value.append([index,Hi,BG_HX_mean])
    # target area colored
    for i in tar_hx_coord:
        row_min = i[0]*8            
        row_max = (i[0]+1)*8-1
        col_min = i[1]*8
        col_max = (i[1]+1)*8-1
        Visual_KP[row_min:row_max+1,col_min:col_max+1,0] = Target_Not_KP_color[0]
        Visual_KP[row_min:row_max+1,col_min:col_max+1,1] = Target_Not_KP_color[1]
        Visual_KP[row_min:row_max+1,col_min:col_max+1,2] = Target_Not_KP_color[2]
    ## shadow area colored
    for i in shadow_hx_coord:
        row_min = i[0]*8            
        row_max = (i[0]+1)*8-1
        col_min = i[1]*8
        col_max = (i[1]+1)*8-1
        Visual_KP[row_min:row_max+1,col_min:col_max+1,0] = Shadow_Not_KP_color[0]
        Visual_KP[row_min:row_max+1,col_min:col_max+1,1] = Shadow_Not_KP_color[1]
        Visual_KP[row_min:row_max+1,col_min:col_max+1,2] = Shadow_Not_KP_color[2]
    cv2.imwrite(os.path.join(save_path,img_HX_path.split('/')[-1].replace('.jpg', '_target_final_mask.jpg')),final_mask)
    cv2.imwrite(os.path.join(save_path,img_HX_path.split('/')[-1].replace('.jpg', '_target.jpg')),Visual_KP)
    # KP in target and background area colored
    for coord in All_KP:
        row_min = coord[0]*8           
        row_max = (coord[0]+1)*8-1
        col_min = coord[1]*8
        col_max = (coord[1]+1)*8-1
        if coord in tar_hx_coord:
            Visual_KP[row_min:row_max+1,col_min:col_max+1,0] = Target_KP_color[0]
            Visual_KP[row_min:row_max+1,col_min:col_max+1,1] = Target_KP_color[1]
            Visual_KP[row_min:row_max+1,col_min:col_max+1,2] = Target_KP_color[2]
        elif coord in shadow_hx_coord:
            Visual_KP[row_min:row_max+1,col_min:col_max+1,0] = Shadow_KP_color[0]
            Visual_KP[row_min:row_max+1,col_min:col_max+1,1] = Shadow_KP_color[1]
            Visual_KP[row_min:row_max+1,col_min:col_max+1,2] = Shadow_KP_color[2]
        else:
            Visual_KP[row_min:row_max+1,col_min:col_max+1,0] = BackGround_KP_color[0]
            Visual_KP[row_min:row_max+1,col_min:col_max+1,1] = BackGround_KP_color[1]
            Visual_KP[row_min:row_max+1,col_min:col_max+1,2] = BackGround_KP_color[2]
    # cv2.imwrite(os.path.join(save_path,img_HX_path.split('/')[-1].replace('.jpg', '_KPvisual_b='+str(b)+'.jpg')),Visual_KP)
    img2 = np.zeros_like(Visual_KP)
    img2[:,:,0] = img
    img2[:,:,1] = img
    img2[:,:,2] = img
    combine_img = cv2.addWeighted(cv2.resize(img2,(128,128)),0.7,cv2.resize(Visual_KP,(128,128)),0.3,0)
    cv2.imwrite(os.path.join(save_path,img_HX_path.split('/')[-1].replace('.jpg', '_KPvisual_b='+str(b)+'_combine.jpg')),combine_img)
def GrayToRGB(img):
    src = img[:,:,0]
    src_gray = img
    
    B = src_gray[:,:,0]
    G = src_gray[:,:,1]
    R = src_gray[:,:,2]
    
    g = src[:]
    p = 0.2989; q = 0.5870; t = 0.1140
    B_new = (g-p*R-q*G)/t
    B_new = np.float64(B_new)
    src_new = np.zeros((src_gray.shape)).astype("float64")
    src_new[:,:,0] = B_new
    src_new[:,:,1] = G
    src_new[:,:,2] = R
    return src_new

if __name__ == '__main__':
    path_noise = 'src/ALL_'
    path_img = 'MSTAR/SOC/img'
    save_path = 'src/list_final'
    target = []
    shadow = []
    lines = os.listdir(path_img)

    noise_path = path_noise+Type
    Target = []
    Shadow = []
    Back = []
    b=0.3
    for img in lines:       
        img_path = os.path.join(path_img,img)
        KP_visual(Type,img_path,b,noise_path,save_path)
