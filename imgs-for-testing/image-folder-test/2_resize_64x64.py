import PIL
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
i = 1
j = 1
foldername = "resize_64x64";
imagesize = 64,64;
for filename in glob.glob("*.jpg"):
    print (filename)
    line_i = os.path.splitext(filename)[0].split("-")
    last = line_i[-1]
    print(line_i)
    size = imagesize
    im=Image.open(filename)
    im = im.resize( size, Image.ANTIALIAS)
    if(line_i[0] == 'abnormal'):
        im.save('./'+foldername+'/abnormal-test-'+str(j)+'.jpg')
    else:
        im.save('./'+foldername+'/normal-test-'+str(j)+'.jpg')
    j = j+1
    
