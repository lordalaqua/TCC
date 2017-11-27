import os
import sys

import numpy as np
import cv2

path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Flip - Error: no image argument passed.")
        sys.exit(1)
    elif (len(sys.argv) < 3):
        image_name = sys.argv[1]
        output_name = image_name
    else:
        image_name = sys.argv[1]
        output_name = sys.agv[2]
    image = cv2.imread(image_name)
    flipped = cv2.flip(image, 1)
    cv2.imwrite(output_name, flipped)
