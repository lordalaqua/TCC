import os
import sys

from numpy import *
from rotatesphere import composeRotationMatrix, eulerFromR

def rotateSphere(image, alpha, beta, gamma):
    # rotate the sphere according alpha, beta, gamma
    (a,b,c) = map(str, (alpha, beta, gamma))
    os.system('python rotatesphere.py ' + image + ' ' + a + ' ' + b + ' ' + c)

def getCrop(image, fov):
    # select and crop the central region of image 
    os.system('octave-cli --eval "getcrop ' + str(fov) + ' ' + image + '" > nul 2>&1')

def depthPrediction(image, output):
    os.system('python ../depth-estimation/run.py ' + image + ' ' + output + ' > nul 2>&1')

def reprojectToSphere(crop, spherical, fov):
    os.system('octave-cli --eval "projectcroptothesphere ' + str(fov) +' ' + spherical + ' ' + crop  + '" > nul 2>&1')

def rotateReprojection(reprojection, alpha, beta, gamma):
    # inverse rotation angles in radians
    a,b,c = eulerFromR(composeRotationMatrix(alpha,beta,gamma).T)
    rotateSphere(reprojection, a, b, c)

if __name__ == "__main__":
    counter = 0
    image = sys.argv[-1]
    crop = 'crop.png'
    reprojection = 'spherical.png'
    validMap = 'validmap.png'
    depth = 'depth.png'
    rotated = 'rotated-' + image
    result = 'rotated-' + reprojection
    resultMap = 'rotated-' + validMap

    horizontalFOV = 90 #60.0
    fov = (90,90)

    angles = [(0,0), (0,90), (0,180), (0,270), (90,0), (-90,0)]
    # phi = vertical angle, theta = horizontal angle
    for phi,theta in angles:
        print(theta, phi)
        current = counter
        print(current, 'of', 5)
        # rotation angles in radians
        alpha, beta, gamma = radians([0, phi, theta])
        
        rotateSphere(image, alpha, beta, gamma, writeTo)
        print(current,  "Image rotated")

        getCrop(rotated, horizontalFOV)
        print(current,  "Image cropped")
        
        # print( current,  "Begin depth prediction...")
        # depthPrediction(crop, depth)
        # print( current,  "Depth calculated")

        # reprojectToSphere(depth, rotated, horizontalFOV)
        # print( current,  "Depth reprojected")

        # rotateReprojection(reprojection, alpha, beta, gamma)
        # print( current,  "Depth rotated back to center")
        # rotateReprojection(validMap, alpha, beta, gamma)
        # print( current,  "Depth rotated back to center")

        # store partial data
        it = str(counter)
        # os.system('mv ' + rotated + ' partial/rotated/' + it + '.jpg')
        # os.system('mv ' + result + ' partial/reprojection/' + it + '.png')
        # os.system('mv ' + resultMap + ' partial/validmap/' + it + '.png')
        os.system('mv ' + crop + ' partial/crops/' + it + '.png')
        # os.system('mv ' + depth + ' partial/depth/' + it + '.png')
        os.system('rm ' + reprojection)
        os.system('rm ' + validMap)            
        counter += 1
         
    
         
    
     #if counter == 1: sys.exit()
         
