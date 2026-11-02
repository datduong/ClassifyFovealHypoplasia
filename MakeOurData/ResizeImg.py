from PIL import Image 
 
# Opens a image in RGB mode 

img_path = ' '
im = Image.open(img_path) 

# Cropped image of above dimension 
# (It will not change original image) 
newsize = (224, 224)
img = im.resize(newsize)
# Shows the image in image viewer 
img.show() 
