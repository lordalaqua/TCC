import glob, os, re

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(SCRIPT_PATH,'dataset')
depth_dir = os.path.join(SCRIPT_PATH,'sphere-cnn')

def depthFayao(image, output):
    return not os.system('matlab -nodisplay -nosplash -nodesktop -r -wait "cd %s; demo_modified %s %s; exit;"' %
                         (os.path.join(SCRIPT_PATH, 'depth-fayao', 'demo'), image, output))

def runPipeline(file, output):
    os.system('python run.py -i %s -o %s' % (file, output))

# # Run depth prediction on equirectangular images
# depthFayao(dataset_dir, depth_dir)

# # Run pipeline for each image
# for file in glob.glob(os.path.join(dataset_dir, "*.png")):
    # output = os.path.basename(file).replace('.png','')
    # runPipeline(file, output)
    