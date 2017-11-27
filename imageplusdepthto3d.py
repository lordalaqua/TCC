import cv2
import sys
import struct
from numpy import *
from scipy import ndimage
from numpy.linalg import norm

if __name__ == '__main__':
    
    if len(sys.argv) < 4: print 'Use python imageplusdepthto3d.py <color image> <depth image> <output filename.pcd>'
    _, colorfname, depthfname, outputfname = sys.argv
    
    color = cv2.imread(colorfname)
    depth = cv2.imread(depthfname, 0)
    
    h, w, _  = color.shape

    phi, theta = meshgrid(linspace(0, pi, num = h, endpoint=False), linspace(0, 2 * pi, num = w, endpoint=False))
    
    depthaux = depth.reshape(-1)
    idx = depthaux != 0 #desconsidera valores de depth = 0

    depth = (stack([(sin(phi) * cos(theta)).T,(sin(phi) * sin(theta)).T,cos(phi).T], axis=2).reshape((w*h,3)).T * depthaux).T
    
    color = color.reshape((w*h,3))
    colors1d = zeros((w*h,1))
    
    for i in range(len(colors1d)): 
        colors1d[i] =  struct.unpack('f', chr(color[i][0]) + chr(color[i][1]) + chr(color[i][2]) + chr(0))[0] 
    
    depth = depth[idx]
    colors1d = colors1d[idx]

    savetxt(outputfname, hstack((depth,colors1d)), header='VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4    4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 1\nHEIGHT '+str(len(depth))+'\nVIEWPOINT 0 0 0 0 1 0 0\nPOINTS '+str(len(depth))+'\nDATA ascii', comments='')
