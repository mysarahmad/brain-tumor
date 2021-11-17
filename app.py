from flask import Flask,request,redirect,render_template
from PIL import Image
import keras
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')




@app.route('/brain_tumor_pred',methods=["GET","POST"])
def brain_tumor_pred():
   #img = request.files['img']
   #img.save('uploads/brain_tumor_img.jpg')
   image = Image.open(request.files['img'])
   model = keras.models.load_model('A:\\deep learning\\brain tumour\\models\\brain_tumor.h5')
   x = np.array(image.resize((100,100)))
   x=x.reshape(1,100,100,3)
   res=model.predict(x)
   if np.argmax(res)==0:
       return render_template("no_tumor.html")
   else:
       return render_template("tumor.html")



if __name__=="__main__":
    app.run(debug=True)