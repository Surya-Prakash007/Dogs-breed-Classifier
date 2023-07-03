import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO , BytesIO
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import zipfile
with zipfile.ZipFile('CNN20.zip', 'r') as zip_ref:
    zip_ref.extractall()

st.title('Dogs Breed Classifier')
pic=Image.open('picture.png')
st.sidebar.image(pic)
st.sidebar.write(":red[Stanford dog breed dataset is used for training the model .]")
st.sidebar.write(":red[Inception_V3 model is used as a base model .]")
st.sidebar.write(":red[Transfer Learning is used while using sequential API of tensorflow.]")

# model = pickle.load(open('cnnmodel10.pkl', 'rb'))
# pickle_in = open("CNN20.pkl", "rb")
# cnn = pickle.load(pickle_in)
li=[ "Chihuahua" ,  "Japanese_spaniel" , "Maltese_dog" , "Pekinese" ,  "Shih-Tzu" ,  "Blenheim_spaniel" , "papillon" , "toy_terrier" , "Rhodesian_ridgeback" , "Afghan_hound" , "basset" , "beagle" , "bloodhound" , "bluetick" , "black-and-tan_coonhound" , "Walker_hound" , "English_foxhound" , "redbone" , "borzoi" , "Irish_wolfhound" , "Italian_greyhound" , "whippet" , "Ibizan_hound" , "Norwegian_elkhound" , "otterhound" , "Saluki" , "Scottish_deerhound" ," Weimaraner" , "Staffordshire_bullterrier" , "American_Staffordshire_terrier" , "Bedlington_terrier" ,  "Border_terrier" ,  "Kerry_blue_terrier" , "Irish_terrier" , "Norfolk_terrier" , "Norwich_terrier" , "Yorkshire_terrier" , "wire-haired_fox_terrier" , "Lakeland_terrier" , "Sealyham_terrier" , "Airedale" , "cairn" , "Australian_terrier" , "Dandie_Dinmont" ," Boston_bull" , "miniature_schnauzer" , "giant_schnauzer" ," standard_schnauzer" , "Scotch_terrier" , "Tibetan_terrier" , "silky_terrier" , "soft-coated_wheaten_terrier" , "West_Highland_white_terrier" , "Lhasa" , "flat-coated_retriever" , "curly-coated_retriever" , "golden_retriever" , "Labrador_retriever" , "Chesapeake_Bay_retriever" , "German_short-haired_pointer" ," vizsla ", "English_setter" , "Irish_setter" ," Gordon_setter "," Brittany_spaniel" , "clumber" , "English_springer" , "Welsh_springer_spaniel" , "cocker_spaniel" , "Sussex_spaniel" , "Irish_water_spaniel" , "kuvasz" , "schipperke" , "groenendael" , "malinois" ," briard" ," kelpie" , "komondor" , "Old_English_sheepdog" , "Shetland_sheepdog" , "collie" , "Border_collie" , "Bouvier_des_Flandres" , "Rottweiler" , "German_shepherd" , "Doberman" , "miniature_pinscher" , "Greater_Swiss_Mountain_dog" , "Bernese_mountain_dog" , "Appenzeller" , "EntleBucher" , "boxer" , "bull_mastiff" , "Tibetan_mastiff" , "French_bulldog" , "Great_Dane" , "Saint_Bernard" , "Eskimo_dog" , "malamute" , "Siberian_husky" , "affenpinscher" , "basenji" , "pug" , "Leonberg" , "Newfoundland" , "Great_Pyrenees" , "Samoyed" , "Pomeranian" , "chow" , "keeshond" , "Brabancon_griffon" , "Pembroke" , "Cardigan" , "toy_poodle" , "miniature_poodle" , "standard_poodle" , "Mexican_hairless" , "dingo" , "dhole" , "African_hunting_dog" ]

from tensorflow.keras.models import load_model
cnn=load_model("CNN20.h5")



uploaded_file = st.file_uploader("Upload a dog image to see its breed" , type=['png','jpg'])
print(type(uploaded_file))
if uploaded_file is not None:

    image1 = uploaded_file.read()
    st.write(":red[Your uploaded image]")
    img1 = st.image(image1,use_column_width=True)
    image = load_img(uploaded_file,target_size=(128,128 , 3))  

    img = img_to_array(image)

    img = np.array(image) 
    # st.image(img)
    img = img / 255.0
    image = img.resize((128,128 , 3))

    img = img.reshape(1 ,128,128 ,-1)
    
    # st.write(img)
    pred=cnn.predict(img)
    # st.write(pred)
    ind=np.argmax(pred)
    # print(li[ind])
    str="The image is of " + li[ind]
    st.header(str )
