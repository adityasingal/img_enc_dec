import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def gen_keys(x,r,size):
    keys=[]
    for _ in range(size):
        x=r*x*(1-x) #logistic map
        keys.append(int((x* pow(10,16))%256))#key=(x*10^16)%256
    print(keys)
    return keys

img = mpimg.imread('poke.bmp')
    
plt.imshow(img)
plt.show()

height = img.shape[0]
width = img.shape[1]
keys = gen_keys(0.011,3.94,height*width)

z = 0
enimg = np.zeros(shape = [height,width,4], dtype = np.uint8)
for i in range(height):
    for j in range(width):
        enimg[i,j] = img[i,j]^keys[z]
        z+=1

plt.imshow(enimg)
plt.show()
plt.imsave('enc_poke.bmp',enimg)

keys = gen_keys(0.010,3.94,height*width)
z = 0
decimg = np.zeros(shape = [height,width,4], dtype = np.uint8)
for i in range(height):
    for j in range(width):
        decimg[i,j] = enimg[i,j]^keys[z]
        z+=1

plt.imshow(decimg)
plt.show()
plt.imsave('dec_poke.bmp',decimg)



import cv2 as cv
img = cv.imread("fame.jpeg") # name of the file we are importing
b, g, r = cv.split(img)
cv.imshow("img", img)

plt.hist(b.ravel(), 256, [0, 256])  
plt.hist(g.ravel(), 256, [0, 256])
plt.hist(r.ravel(), 256, [0, 256])

hist2 = cv.calcHist([img], [2], None, [256], [0, 256])
hist1 = cv.calcHist([img], [1], None, [256], [0, 256])
hist = cv.calcHist([img], [0], None, [256], [0, 256])
#exposure histogram
plt.plot(hist)
plt.plot(hist1)
plt.plot(hist2)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
