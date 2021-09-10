# To work in Jupyter - as part of https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass/
# img is the directory path of the image you want to quantise
# num is the number of different colours you want to use

def quant_col(img, num):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from sklearn.cluster import KMeans

    #image as array
    image_as_array = mpimg.imread(img)
    (h,w,c) = image_as_array.shape
    image_as_2d = image_as_array.reshape(h*w,c)

    #quantise colours
    model = KMeans(n_clusters=num)
    labels = model.fit_predict(image_as_2d)
    rgb_codes = model.cluster_centers_.round(0).astype(int)
    quantised_img = np.reshape(rgb_codes[labels], (h,w,c)) # reshape back to 3D

    #display
    plt.figure(dpi = 200)
    plt.imshow(quantised_img)
