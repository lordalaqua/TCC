from numpy import *

from glob import glob
import cv2

fact = .5
count = 0
threshold = 250
for (reprojection,validmap) in zip(glob('partial/reprojection/*.png'), glob('partial/validmap/*.png')):
    im=cv2.imread(reprojection)
    validPixels = cv2.imread(validmap)
    # im = cv2.resize(im,(0,0),fx=fact,fy=fact)
    if count == 0: 
        final = im
    else: 
        #final += im
        for i in range(final.shape[0]):
            for j in range(final.shape[1]):
                if (validPixels.item(i,j,0) == 255):
                    pixel = [im.item(i,j,c) for c in range(3)]
                    for c in range(3):
                        final.itemset((i,j,c),pixel[c])
                    # if (not (isclose(final[i,j],zeros(3),0)).all()):
                    #     final[i,j] = final[i,j] / 2
                        
        #final[...,0] += where(final[...,0] > 0, im[...,0], -final[...,0])
        #final[...,1] += where(final[...,1] > 0, im[...,1], -final[...,1])
        #final[...,2] += where(final[...,2] > 0, im[...,2], -final[...,2])

    print (count)
    count += 1

cv2.imwrite('output.png',final)

