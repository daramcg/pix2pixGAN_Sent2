from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from matplotlib import pyplot
from numpy import vstack
import cv2

#I changed this to model I generated
#note remove the compile=False option if making new model

path="C:/Users/daram/pix2pixGAN_Sent2/maps/Sentinel2/"
filename="Test3.jpg"


model = load_model('model_000100.h5',compile=False)




# load and resize the image
pixels = load_img(path + filename)
# convert to numpy array
pixels = img_to_array(pixels)
src_image = pixels



# plot source and generated images
def plot_images(src_img, gen_img):
    images = vstack((src_img, gen_img))
    # scale from [-1,1] to [0,1]
    images = (images+1) / 2.0
    titles = ['Source', 'Generated']
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 2, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show title
        pyplot.title(titles[i])
    pyplot.show()

# So I can plot it
image_to_plot = src_image
image_to_plot = cv2.resize(image_to_plot, (256, 256))
plotMeLater = image_to_plot.reshape(1, 256, 256, 3)


#Resize for the model:
src_image = cv2.resize(src_image, (256, 256))     # resize image to match model's expected sizing
src_image = src_image.reshape(1, 256, 256, 3) # return the image with shape model wants

# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(plotMeLater, gen_image)
