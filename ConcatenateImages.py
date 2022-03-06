import cv2
import numpy as np

pathSat = r"C:/Users/daram/GY642Files/Project/ExperimentTrainingData_sat/"
pathTifs = r"C:/Users/daram/GY642Files/Project/ExperimentTrainingData_boundary/"


range = np.arange(1,11,1)

for i in range:
    for j in range:
        print(i, j)
        img1 = cv2.imread(pathSat+"subsets_" + str(j)+"_"+ str(i) +".tif")
        img2 = cv2.imread(pathTifs + "boundary_" + str(j) + "_" + str(i) + ".tif")
        # resize for hconcat
        img1 = cv2.resize(img1, (600, 600))
        img2 = cv2.resize(img2, (600, 600))
        # use guassian enhancement
        # img1 = cv2.GaussianBlur(img1, (3, 3), 0)
        # horizontally concatenates images of same height
        im_h = cv2.hconcat([img1, img2])
        # show the output image
        cv2.imwrite("C:/Users/daram/pix2pixGAN_Sent2/maps/test/train_" + str(j) + "_"+ str(i) +".tif", im_h)


