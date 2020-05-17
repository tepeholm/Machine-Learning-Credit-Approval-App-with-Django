import time
import base64

#from PIL import Image
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import render

from .forms import form_canvas
from .models import count

#from io import BytesIO
from matplotlib import image
import tensorflow as tf
import cv2
import numpy as np


def main(request):
    if request.method == 'POST':
        form = form_canvas(request.POST)
        tim = request.POST.get('im')
        #o = count(name='image_yzat')
        #o.save()
        if form.is_valid():
            data = base64.b64decode(tim)          
            print(type(data))
            print('***************************************************************')

            img0 = "ml_app/test.bmp"
            #img0 = r"C:\Users\Yzat\Desktop\publishto heroku\Credit_Approval\Credit_project\ml_app\test.bmp"
        
            with open(img0, 'wb') as f:
                f.write(data)
                f.close
                
            o = count(name=img0)
            o.save()
            

            print(type(img0))
            print('+++++++++++++++++++++++++++++++++++')
            
            # to save image in different files
            '''
            img1 = Image.open(img0)
            idi = o.id
            a = "C:/Users/Yzat/Downloads/ML_Django/Credit_Approval/Credit_project/media/"
            adr = a + str(idi) + '.bmp'
            img1.save(adr)
            '''

            ####################    ML Model  ##################################
            new_model = tf.keras.models.load_model('ml_app/yzat.h5')
            img = cv2.imread('ml_app/test.bmp', -1)
            b, g, r, alpha = cv2.split(img)
            img_BGR = cv2.merge((r, g, alpha))
            image = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))
            image = image.astype('float32')
            image = image.reshape(1, 28, 28, 1)
            image /= 255
            predictions = new_model.predict(image)
            yzat_predictions = np.argmax(predictions)
            print('******************', yzat_predictions, '**********************') 
            messages.success(request,'{}'.format(yzat_predictions))
            
            #################### ML Model End ####################################
            
            return HttpResponseRedirect('/canvas')
        else:
            
            return HttpResponseRedirect('/canvas')

    else:
        return render(request, 'canvas.html')
