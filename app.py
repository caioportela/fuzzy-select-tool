import matplotlib.pyplot as plt
from skimage import color, io
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, flood

def main():
    img = io.imread('interior1.jpg')  # Open image

    img_hsv = color.rgb2hsv(img)  # Convert color channels
    img_gray = color.rgb2gray(img)  # Convert color channels

    # Sobel edge detection
    edge_sobel = sobel(img_gray)

    # Felzenszwalb’s image segmentation
    segments_fz = felzenszwalb(edge_sobel, scale=100, sigma=1, min_size=150)

    # Create a mask of flooded pixels
    mask = flood(segments_fz, (400, 1600), tolerance=0.5)  # interior1
    # mask = flood(segments_fz, (130, 700), tolerance=0.5)  # interior2
    # mask = flood(segments_fz, (90, 450), tolerance=0.5)  # interior3
    # mask = flood(segments_fz, (130, 700), tolerance=0.5)  # interior4

    # Set pixels of mask to new value for hue channel
    img_hsv[mask, 0] = 0.15

    # Set pixels of mask to new value for saturation channel
    img_hsv[mask, 1] = 0.8

    # Setup image plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 10))

    ax[0].imshow(img)
    ax[0].set_title('Original')

    ax[1].imshow(color.hsv2rgb(img_hsv))
    ax[1].set_title('Customized')

    plt.show()

if __name__ == '__main__':
    main()
