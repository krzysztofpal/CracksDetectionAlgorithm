import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (127, 127, 0),
    7: (127, 0, 127),
    8: (0, 127, 127),
    9: (255, 127, 0),
    10: (255, 0, 127),
    11: (127, 255, 0),
    12: (0, 255, 127),
    13: (127, 0, 255),
    14: (0, 127, 255)
}

def plotComponentsAnalysisResults(results, filter_fn=None, drawObject=None):
    numLabels, labels, stats, centroids = results
    regions_selected = np.zeros((labels.shape[0], labels.shape[1], 3))
    
    for i in range(1, numLabels):
        
        if filter_fn is not None and filter_fn(stats[i]) == False:
            continue
        
        color = colors[i % len(colors)]
        region = np.ones_like(regions_selected) * color
        region = cv.bitwise_and(region, region, mask=(labels == i).astype(np.uint8) * 255)

        regions_selected = regions_selected + region

    if drawObject is None:
        plt.imshow(regions_selected)
    else:
        drawObject.imshow(regions_selected)


def compute(img, width=256, size_limit = 8, output_plots_path: str = None):
    

    th =  cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,2)
    ret2,th2 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.bitwise_not(th2)
    kernel = np.ones((5, 5), np.uint8)
    th3 = cv.erode(th2, kernel, iterations=4)
    th4 = cv.dilate(th3, kernel, iterations=6)
    th5 = cv.bitwise_not(th4)
    th6 = cv.bitwise_and(th, th5)
    height = int((img.shape[0] / img.shape[1]) * width)
    dim = (width, height)
    resized = cv.resize(th6, dim, interpolation = cv.INTER_AREA)
    resized = cv.threshold(resized,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    

    # connectivity components analysis
    connectedComponents = cv.connectedComponentsWithStats(resized, 8, cv.CV_32S)
    stats = connectedComponents[2]
    
    toreturn = []

    # nanieś znalezione krawędzie na oryginalny obraz
    rimg = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    fractures = np.zeros_like(rimg)

    labels = connectedComponents[1]
    for i in range(1, connectedComponents[0]):
        area = stats[i][-1]
        if area <= size_limit:
            continue

        mask = np.asarray(labels == i, np.uint8)
        mask = cv.resize(mask, (img.shape[1], img.shape[0]))
        mask = cv.threshold(mask,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        mask = cv.bitwise_and(th6, mask)
        toreturn.append(mask)

        color = colors[i % len(colors)]
        region = np.ones_like(rimg) * color
        region = cv.bitwise_and(region, region, mask=mask)
        fractures = fractures + region

    

    if output_plots_path is not None:
        rr = rimg.reshape(-1, 3)
        rr[fractures.sum(axis=-1).reshape(-1) > 0] = fractures.reshape(-1,3)[fractures.sum(axis=-1).reshape(-1) > 0].reshape(-1,3)
        rr = rr.reshape(rimg.shape[0], rimg.shape[1], 3)
        
        def sf(name):
            plt.savefig(os.path.join(output_plots_path, f"{name}.png"))
            plt.clf()

        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        sf("1. Original Image")

        plt.imshow(th, cmap='gray')
        plt.title("Adaptive Gaussian Threshold")
        sf("2. Adaptive Gaussian Threshold")

        plt.imshow(th6, cmap="gray")
        plt.title("Filtered out bubbles and shadows")
        sf("3. Filtered out bubbles and shadows")

        plt.imshow(resized, cmap='gray')
        plt.title("Resized Adaptive Gaussian Threshold")
        sf("4. Resized Adaptive Gaussian Threshold")

        plotComponentsAnalysisResults(connectedComponents, drawObject=None)
        plt.title("Connected Component Analysis")
        sf("5. Connected Components Analysis results")

        plotComponentsAnalysisResults(connectedComponents, filter_fn=lambda s: s[-1] >= size_limit, drawObject=None)
        plt.title("Connected Components Analysis small regions filter")
        sf("6. Connected Components Analysis small regions filter")

        plt.imshow(fractures)
        plt.title("Detected fractures upscaled")
        sf("7. Detected fractures upscaled")

        plt.imshow(rr)
        plt.title("Detected fractures on original image")
        sf("8. Detected fractures on original image")

    return toreturn

