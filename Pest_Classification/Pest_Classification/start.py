from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
from flask import send_file

from numpy import *
import cv2

import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model = load_model('pest.keras', custom_objects=custom_objects)

from info import (
    killer_bees_data, anaphe_reticulata_data, anomis_sabulifera_data, ants_data, aphids_data,
    apion_corchori_data, armyworms_data, beet_armyworm_data, beetles_data, black_hairy_data,
    bollworm_data, brown_marmorated_stink_bugs_data, cabbage_loopers_data, caterpillar_pests_data,
    citrus_canker_data, colorado_potato_beetles_data, corn_borer_data, corn_earworms_data,
    cutworms_data, earwig_data, fall_armyworms_data, field_cricket_data, flea_beetle_data,
    fruit_flies_data, gall_midge_data, grasshopper_data, grub_data, hornworms_data, jute_hairy_data,
    leaf_beetle, leafhoppers, mealybugs, mirid_bugs, mosquitoes, moths, pod_borers, red_mites,
    sawfly, Scopula_Emissaria, slugs, snails, spider_mites, stem_maggots, stem_borers, termites,
    Termite_odontotermes, thrips, wasps, weevils, western_corn_rootworms, wireworms, yellow_mites
)

#Mappings

pest_dictionary = {
    'Africanized Honey Bees (Killer Bees)': killer_bees_data,
    'Anaphe reticulata': anaphe_reticulata_data,
    'Anomis sabulifera': anomis_sabulifera_data,
    'Ants': ants_data,
    'Aphids': aphids_data,
    'Apion corchori': apion_corchori_data,
    'Armyworms': armyworms_data,
    'Beet Armyworm': beet_armyworm_data,
    'Beetle': beetles_data,
    'Black Hairy': black_hairy_data,
    'Bollworm': bollworm_data,
    'Brown Marmorated Stink Bugs': brown_marmorated_stink_bugs_data,
    'Cabbage Loopers': cabbage_loopers_data,
    'Catterpillar': caterpillar_pests_data,
    'Citrus Canker': citrus_canker_data,
    'Colorado Potato Beetles': colorado_potato_beetles_data,
    'Corn borer': corn_borer_data,
    'Corn Earworms': corn_earworms_data,
    'Cutworm': cutworms_data,
    'Earwig': earwig_data,
    'Fall Armyworms': fall_armyworms_data,
    'Field Cricket': field_cricket_data,
    'Flea beetle': flea_beetle_data,
    'Fruit Flies': fruit_flies_data,
    'Gall midge': gall_midge_data,
    'Grasshopper': grasshopper_data,
    'Grub': grub_data,
    'Hornworms': hornworms_data,
    'Jute Hairy': jute_hairy_data,
    'Leaf Beetle': leaf_beetle,
    'Leafhopper': leafhoppers,
    'Mealybug': mealybugs,
    'Miridae': mirid_bugs,
    'Mosquito': mosquitoes,
    'Moth': moths,
    'Pod Borer': pod_borers,
    'Red Mite': red_mites,
    'Sawfly': sawfly,
    'Scopula Emissaria': Scopula_Emissaria,
    'Slug': slugs,
    'Snail': snails,
    'Spider Mites': spider_mites,
    'Stem maggot': stem_maggots,
    'Stem_borer': stem_borers,
    'Termite': termites,
    'Termite odontotermes': Termite_odontotermes,
    'Thrips': thrips,
    'Wasp': wasps,
    'Weevil': weevils,
    'Western Corn Rootworms': western_corn_rootworms,
    'Wireworm': wireworms,
    'Yellow Mite': yellow_mites
}

classes = ['Africanized Honey Bees (Killer Bees)',
 'Anaphe reticulata',
 'Anomis sabulifera',
 'Ants',
 'Aphids',
 'Apion corchori',
 'Armyworms',
 'Beet Armyworm',
 'Beetle',
 'Black Hairy',
 'Bollworm',
 'Brown Marmorated Stink Bugs',
 'Cabbage Loopers',
 'Catterpillar',
 'Citrus Canker',
 'Colorado Potato Beetles',
 'Corn borer',
 'Corn Earworms',
 'Cutworm',
 'Earwig',
 'Fall Armyworms',
 'Field Cricket',
 'Flea beetle',
 'Fruit Flies',
 'Gall midge',
 'Grasshopper',
 'Grub',
 'Hornworms',
 'Jute Hairy',
 'Leaf Beetle',
 'Leafhopper',
 'Mealybug',
 'Miridae',
 'Mosquito',
 'Moth',
 'Pod Borer',
 'Red Mite',
 'Sawfly',
 'Scopula Emissaria',
 'Slug',
 'Snail',
 'Spider Mites',
 'Stem maggot',
 'Stem_borer',
 'Termite',
 'Termite odontotermes',
 'Thrips',
 'Wasp',
 'Weevil',
 'Western Corn Rootworms',
 'Wireworm',
 'Yellow Mite']


@app.route('/')
def index():
    return render_template('welcome.html')


@app.route('/open')
def index2():
    return render_template('start.html')

@app.route('/start')
def start():
    return render_template('image.html')

file_path_g1 = ""


from flask import render_template

@app.route('/upload', methods=['GET', 'POST'])
def upload_detection1():
    global file_path_g1
    
    if request.method == 'POST':
        f1 = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path1 = os.path.join(basepath, 'uploads', secure_filename(f1.filename))
        f1.save(file_path1)

        file_path_g1 = file_path1

        return render_template("message.html", image_path=file_path1)

    return "None"


@app.route('/get_uploaded_image')
def get_uploaded_image():
    global file_path_g1
    # Extract the file extension
    _, file_extension = os.path.splitext(file_path_g1)
    
    # Determine the mimetype based on the file extension
    if file_extension.lower() == '.jpg' or file_extension.lower() == '.jpeg':
        mimetype = 'image/jpeg'
    elif file_extension.lower() == '.png':
        mimetype = 'image/png'
    elif file_extension.lower() == '.bmp':
        mimetype = 'image/bmp'
    else:
        # Set a default mimetype for unknown file types
        mimetype = 'image/jpeg'
    
    # Return the uploaded image file
    return send_file(file_path_g1, mimetype=mimetype)



g_output = ""

def pred(image_name):
    
    #load the image
    img= cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize the image
    resize=cv2.resize(img,(256,256))
   

    #optimize the new image
    resize=resize/255

    
    #expand your image array
    img=expand_dims(resize,0)


    predictions = model.predict(img)

    print(predictions,"*************")
    

    # Convert the predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    print(predicted_labels,"##############")

    global g_output

    dictonary = str(classes[predicted_labels[0]])
    g_output = pest_dictionary[dictonary]



    value = "The Pest detected in Your Agriculture Field is: "+str(classes[predicted_labels[0]])

    print(value)
    
    return value
   




@app.route('/predict')
def predict():

    output = pred(file_path_g1)

    return render_template('output.html', output_line=output)
   
@app.route('/information')
def information():
    return render_template('desc.html')


# Define routes to render each HTML page
@app.route('/information1')
def information1():

    global g_output
    return render_template('information.html', value="Pest Description", data=g_output["Pest Description"])


# Define routes to render each HTML page
@app.route('/information2')
def information2():

    global g_output
    return render_template('information.html', value="Seasonal Behaviour", data=g_output["Seasonal Behaviour"])



# Define routes to render each HTML page
@app.route('/information3')
def information3():

    global g_output
    return render_template('information.html', value="Crop Damage Effects", data=g_output["Crop Damage Effects"])


# Define routes to render each HTML page
@app.route('/information4')
def information4():

    global g_output
    return render_template('information.html', value="Organic Fertilizer and Pest Prevention", data=g_output["Organic Fertilizer and Pest Prevention"])

# Define routes to render each HTML page
@app.route('/information5')
def information5():

    global g_output
    return render_template('information.html', value="Pest Control Duration", data=g_output["Pest Control Duration"])

# Define routes to render each HTML page
@app.route('/information6')
def information6():

    global g_output
    return render_template('information.html', value="Pest Repetition Chances", data=g_output["Pest Repetition Chances"])


# Define routes to render each HTML page
@app.route('/information7')
def information7():

    global g_output
    return render_template('information.html', value="Natural Predators and Biological Control", data=g_output["Pest Repetition Chances"])


# Define routes to render each HTML page
@app.route('/information8')
def information8():

    global g_output
    return render_template('information.html', value="Weather Effects on Pest Dynamics", data=g_output["Weather Effects on Pest Dynamics"])


# Define routes to render each HTML page
@app.route('/information9')
def information9():

    global g_output
    return render_template('information.html', value="Land Nutrient Effects on Pest and Crop Health", data=g_output["Land Nutrient Effects on Pest and Crop Health"])







if __name__ == '__main__':
    app.run(debug=False)
