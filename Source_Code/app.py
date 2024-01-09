from flask import Flask, render_template, request, send_from_directory, send_file
from keras.models import load_model
import numpy as np
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score
import os
import math
from PIL import Image

app = Flask(__name__, static_folder='public')
app.config['UPLOAD_FOLDER'] = app.static_folder+'/static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
target_img ="public/static/images"

model = load_model("model3.h5", compile=False)  # Ganti dengan path yang benar
model.compile(optimizer="Adam")

@app.route('/')
def index_view():
    return send_from_directory(app.static_folder,'index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((240, 240))  # Sesuaikan ukuran dengan model
    image = np.expand_dims(image, axis=0) #menambah dimensi baru dengan posisi indeks 0
    image = image / 255.0  # Normalisasi gambar jika diperlukan
    
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    
    class_labels = ["Antraknosa", "Embun Jelaga", "Gloeosporium","Kutu Putih"]  # Ganti dengan label kelas Anda
    predicted_class_label = class_labels[predicted_class_index]
    predicted_value = prediction[0][predicted_class_index]
    
    return predicted_class_label, predicted_value


@app.route('/hasil', methods=['GET','POST'])
def hasil():
    if request.method == 'POST':
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            predicted_class_label, predicted_value = predict_image(filepath)
            
            return render_template('hasil.html', jenis_penyakit=predicted_class_label, 
                nilai_akurasi=str(round(100*np.max(predicted_value),3)), 
                nama_file=filename)
            os.remove(filepath)
        
        return "File type not allowed"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
