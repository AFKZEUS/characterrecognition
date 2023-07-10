
from keras.preprocessing.image import load_img
import os
import numpy as np

from keras.models import load_model

model = load_model('smodel.h5')

image_number=1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        image = load_img('digits/digit{}.png'.format(image_number), target_size=(28, 28))
        image = image.convert(mode='L')
        img = np.array(image)
        img = img / 255.0
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.show()

        img = img.reshape(1, 28, 28, 1)

        label = model.predict(img)
        label=np.round(label).astype(int)

        word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X', 34:'Y',35:'Z'}
        z = 0


        for i in label[0]:
             if i==1:
                break
             z+=1


         # print(label)
        print("Input Character is probably :' ",word_dict[z],"'")
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1

