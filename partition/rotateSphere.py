'''
python rotatesphere.py image alpha beta gamma
'''

import cv2
import sys
from numpy import *
from numpy.linalg import norm, inv, det

from scipy.optimize import least_squares, root, minimize, leastsq
from scipy.io import loadmat, savemat

from time import time
from numpy import *
from scipy import ndimage

set_printoptions(precision=5)


def Rx(alpha):
    return matrix([[1,          0,           0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha),  cos(alpha)]])


def Ry(beta):
    return matrix([[cos(beta),  0,   sin(beta)], [0,          1,           0], [-sin(beta), 0,  cos(beta)]])


def Rz(gamma):
    return matrix([[cos(gamma), -sin(gamma), 0], [sin(gamma),  cos(gamma), 0], [0,              0,       1]])


def composeRotationMatrix(alpha, beta, gamma):
    return dot(Rz(gamma), dot(Ry(beta), Rx(alpha)))


def eulerFromR(R, singularThreshold=1e-6):
    singularIdx = sqrt(R[2, 1]**2 + R[2, 2]**2)
    singular = singularIdx < singularThreshold
    if not singular:
        alpha, beta, gamma = arctan2(
            R[2, 1], R[2, 2]), arctan2(-R[2, 0], singularIdx), arctan2(R[1, 0], R[0, 0])
    else:
        alpha, beta, gamma = arctan2(-R[1, 2], R[1, 1]
                                     ), arctan2(-R[2, 0], singularIdx), 0
    return alpha, beta, gamma


class SphericalImage(object):

    def __init__(self, equImage):
        self.__colors = equImage
        self.__dim = equImage.shape  # height and width

        phi, theta = meshgrid(linspace(0, pi, num=self.__dim[0], endpoint=False), linspace(
            0, 2 * pi, num=self.__dim[1], endpoint=False))
        self.__coordSph = stack(
            [(sin(phi) * cos(theta)).T, (sin(phi) * sin(theta)).T, cos(phi).T], axis=2)

    def rotate(self, R):
        data = array(dot(self.__coordSph.reshape(
            (self.__dim[0] * self.__dim[1], 3)), R))
        self.__coordSph = data.reshape((self.__dim[0], self.__dim[1], 3))

        x, y, z = data[:, ].T

        phi = arccos(z)
        theta = arctan2(y, x)
        theta[theta < 0] += 2 * pi
        theta = self.__dim[1] / (2 * pi) * theta
        phi = self.__dim[0] / pi * phi

        if len(self.__dim) > 2:
            for ch in range(self.__dim[2]):
                self.__colors[..., ch] = ndimage.map_coordinates(self.__colors[:, :, ch], [
                                                                 phi, theta], order=1, prefilter=False, mode='reflect').reshape(self.__dim[0], self.__dim[1])

    def getEquirectangular(self): return self.__colors

    def getSphericalCoords(self): return self.__coordSph


if __name__ == '__main__':
    im = cv2.imread(sys.argv[-4])
    im = SphericalImage(im)
    R = composeRotationMatrix(
        (float(sys.argv[-3])), (float(sys.argv[-2])), (float(sys.argv[-1])))
    im.rotate(R)
    cv2.imwrite('rotated-' + sys.argv[-4], im.getEquirectangular())


def rotateSphere(image, alpha, beta, gamma, writeToFile=None, use_mat=False):
    if use_mat:
        data = loadmat(image)
        data = data['data_obj']
        data = data.reshape((data.shape[0], data.shape[1], 1))
    else:
        data = cv2.imread(image)
    sphere = SphericalImage(data)
    sphere.rotate(composeRotationMatrix(alpha, beta, gamma))
    if(writeToFile):
        eq = sphere.getEquirectangular()
        if use_mat:
            eq = eq.reshape((eq.shape[0], eq.shape[1]))
            savemat(writeToFile, {'data_obj': eq})
        else:
            cv2.imwrite(writeToFile, eq)
        return writeToFile
    else:
        return sphere.getEquirectangular()


def rotateBack(image, alpha, beta, gamma, writeToFile=None, use_mat=False):
    a, b, c = eulerFromR(composeRotationMatrix(alpha, beta, gamma).T)
    return rotateSphere(image, a, b, c, writeToFile, use_mat)
