from io import BytesIO

import matplotlib.pyplot as plt
from flask import Flask, request, send_file
from skimage import color, io
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, flood

app = Flask(__name__)

@app.route('/paint', methods=['POST'])
def paint():
    """Return the picture with wall color changed."""

    file = request.files['image']  # Get image from request

    coord_x = int(request.form.get('coord_x'))  # Get X coordinate from request
    coord_y = int(request.form.get('coord_y'))  # Get Y coordinate from request

    img = io.imread(file)  # Open image

    img_hsv = color.rgb2hsv(img)  # Convert color channels
    img_gray = color.rgb2gray(img)  # Convert color channels

    # Sobel edge detection
    edge_sobel = sobel(img_gray)

    # Felzenszwalb’s image segmentation
    segments_fz = felzenszwalb(edge_sobel, scale=100, sigma=1, min_size=150)

    # Create a mask of flooded pixels
    mask = flood(segments_fz, (coord_y, coord_x), tolerance=0.5)

    # Set pixels of mask to new value for hue channel
    img_hsv[mask, 0] = 0.15

    # Set pixels of mask to new value for saturation channel
    img_hsv[mask, 1] = 0.8

    # Convert color channels
    img_final = color.hsv2rgb(img_hsv)

    # Create a byte object to save the image
    file_object = BytesIO()

    # Save image on the byte object
    io.imsave(file_object, img_final, plugin='pil')
    file_object.seek(0)

    return send_file(file_object, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
