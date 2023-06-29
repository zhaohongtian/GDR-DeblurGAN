import cv2

import os
import numpy as np
from skimage import io, data


def ComputeDarkChannel(Input,Size):

    sp = Input.shape

    sz1 = sp[0]  # the row of image

    sz2 = sp[1]  # the col of  image

    sz3 = sp[2] # the num of  the image  a = cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_REPLICATE)

    #Input = np.pad

    Input = cv2.copyMakeBorder(Input,Size,Size,Size,Size, cv2.BORDER_REPLICATE)

    Output = np.zeros((sz1, sz2))

    res_list = []

    #for j in range(len(martix[0])):

    #    one_list = []

    #    for i in range(len(martix)):
    #        one_list.append(int(martix[i][j]))

    #    res_list.append(str(max(one_list)))

    #return res_list

    for i in range(0,sz1 - 1):
        for j in range(0,sz2 - 1):
            patch = Input[i:(i+Size-1),j:(j+Size-1),:]
            # patch2 = np.zeros((Size, Size))
            # for ii in range(0,Size - 1):
            #     one_list = []
            #     for jj in range(0,Size - 1):
            #         patch2[jj][ii] = min(patch[ii,jj,0],patch[ii,jj,1],patch[ii,jj,2])
            #         one_list.append(float(patch2[jj][ii]))
            #     res_list.append(min(one_list))
            # Output[i,j] = min(res_list)
            # res_list = []
            Output[i,j] = patch.min()
            # tmp1 = patch[:,:,0].min()
            # tmp2 = patch[:,:,2].min()
            # tmp3 = patch[:,:,1].min()
            # Output[i, j] = min(tmp1,min(tmp2,tmp3))
    return Output


if __name__ == '__main__':

    Im = cv2.imread('C:\\Users\warre\Desktop\\blurtextimages\sharpimage\\real_img2_result.png')
    res = ComputeDarkChannel(Im, 35)
    print (res)
    #res = round(res,int)

    # cv2.imshow('Res',res)
    # cv2.imshow('Im',Im)
    # cv2.waitKey(0)
    cv2.imwrite('real_img2_result_RES.png', res)
    #cv2.imwrite(res,'C:\\Users\warre\Desktop\\blurtextimages\sharpimage\\real_img2_RES.bmp')




