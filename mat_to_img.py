import argparse
from scipy.io import loadmat
import cv2
import numpy as np

def mapImage(function, image):
    mapped = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image.item((i, j))
            mapped.itemset((i, j), function(pixel, i, j))
    return mapped

def imwriteNormalize(filename, data, color):
    mask = data > 0.001
    maxDiff = np.amax(data[mask])
    minDiff = np.amin(data[mask])
    normalized = mapImage(lambda p, i, j: (
        p - minDiff) / (maxDiff - minDiff) * 255, data)
    if(color):
        normalized =  cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_RAINBOW)
        normalized = cv2.bitwise_and(normalized,normalized,mask = mask.astype(np.uint8))
    cv2.imwrite(filename, normalized)

def loadArray(filename):
    mat = loadmat(filename)
    return mat['data_obj']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate equirectangular image depth.')
    parser.add_argument('mat', metavar='mat')
    parser.add_argument('out', metavar='output', nargs='?', default='mat.jpg')
    parser.add_argument('-color', action='store_true')

    args = parser.parse_args()
    # Get config values from args
    mat = args.mat
    output = args.out
    colormap = args.color

    data = loadArray(mat)
    imwriteNormalize(output,data,colormap)