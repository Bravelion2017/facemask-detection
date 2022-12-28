import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import cv2, os
from skimage import io
from skimage.io import imsave
import io as IO
import base64
import numpy as np
from keras.models import load_model
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import urllib.request

urllib.request.urlretrieve(
        'https://facemaskoseme.s3.amazonaws.com/model_xception_facemask.h5', 'model_xception_facemask.h5')


img_size = (299, 299)
style = {'textAlign': 'center'}
style2 = {'textAlign': 'center', 'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}
steps = 0.1
marks = lambda min, max: {i: f"{i}" for i in range(min, max)}

preprocess_input = tf.keras.applications.xception.preprocess_input  #xception.preprocess_input
test_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)

def imgs_to_df(path, images_list):
    test_imgs = pd.DataFrame(data=images_list, columns=['images'])

    return test_imgs


def facemask(image_df,column_name, image_generator,model,get_key, class_dict, batch_size=16):
    ''':image_df : Should be in Dataframe
    :column_name : name of image column in string
    :image_generator : Tensorflow image generator with preprocessing function
    :model : saved model
    :get_key : function to check np.argmax value then return key
    :class_dict : dictionary for classes
    Image size is (299,299) for Xception Model
    '''
    tensor= image_generator.flow_from_dataframe(image_df, x_col=column_name,
                                         target_size=img_size, class_mode=None,
                                         batch_size=batch_size, shuffle=False)
    new_pred = model.predict(tensor)
    prediction = []
    for i in range(len(new_pred)):
        prediction.append(np.argmax(new_pred[i]))
    pred_df = pd.DataFrame(prediction, columns=['label'])
    pred_df.label = pred_df.label.map(lambda x: get_key(x, class_dict))

    return pred_df.label
def get_key(value, dictionary):
  for key, val in dictionary.items():
    if value == val:
      return key


model= keras.models.load_model('model_xception_facemask.h5')
# ===========================================

my_app = dash.Dash(__name__,external_stylesheets=[dbc.themes.SOLAR])


my_app.layout = html.Div([
    html.H1('Face Mask Detection', style=style), html.Br(),
    html.P("Image Path Input:"),
    dcc.Input(id='imagein',type='text',placeholder='Input image path in quote..'),
    html.Br(),

    html.Br(),
    html.Div(id='my-out')

],style=style)
@my_app.callback(
    Output('my-out','children'),
    [Input('imagein','value')]
)
def update(image):
    if image == '':
        return f"No Image"
    try:
        Image= io.imread(image)
        imsave('testdash.png',Image)
    except:
        return f"No Image"

    # labels
    class_dict = {'With Mask': 0, 'Without Mask': 1}

    # My Test
    try:
        new_path = 'testdash.png'
        imgs = [new_path]
        test_imgs = imgs_to_df(path=new_path, images_list=imgs)
        new_predictions = facemask(test_imgs, 'images', test_gen, model, get_key, class_dict)
    except:
        return f"No Image"

    #Image Try
    arrayed= cv2.imread(new_path)
    fig = px.imshow(arrayed)
    buf = IO.BytesIO()  # in-memory files
    plt.imshow(arrayed)
    plt.axis('off')
    plt.savefig(buf, format="png")  # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
    final= "data:image/png;base64,{}".format(data)
    #End

    # return html.Div([dcc.Graph(figure=html.Img(src=final)),html.Br(),html.H3(new_predictions[0])])
    return html.Div([html.Img(src=final), html.Br(), html.H2(new_predictions[0],style=style)],style=style)




# Please Note: host='127.0.0.1' works for me else host='0.0.0.0' Thank you.
if __name__ == '__main__':
    my_app.run_server(
        port = random.randint(8000,9999),
        host="127.0.0.1"
    )
