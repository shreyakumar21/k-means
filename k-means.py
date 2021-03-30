import numpy as np
import cv2 as cv

def K_Means(Image, K):
    
    if(len(Image.shape)<3):
      Z = Image.reshape((-1,1))
    elif len(Image.shape)==3:
      Z = Image.reshape((-1,3))
    
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    Clustered_Image = res.reshape((Image.shape))
    
    return Clustered_Image

def main():
    Input_Image = cv.imread("ROI-2.jpg")
    Clusters = 5
    Clustered_Image = K_Means(Input_Image, Clusters)
    
    cv.imwrite("ROI2-5.jpg", Clustered_Image)
    input("Prcoessing done!")

if __name__ == '__main__':
	main()
