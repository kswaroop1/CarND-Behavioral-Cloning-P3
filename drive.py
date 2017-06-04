import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2
import math
from keras.models import load_model
import sys
import os
import tensorflow as tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# image preprocessing, same as used while training
def preprocessImage(image, new_size_col, new_size_row):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col, new_size_row), interpolation=cv2.INTER_AREA) 
    return image

# Throttle adjustment
def adjust_throttle_for_speed(speed, min_speed=10.0, max_speed=30.0, boost=10.0):
    if speed < min_speed: return (min_speed+boost-speed)/max_speed
    if speed > max_speed: return (max_speed-boost-speed)/max_speed
    return 0.

def adjust_throttle_for_steering(steering, new_steering):
    return -abs(steering/25.0-new_steering)/0.5

throt,mins,maxs,boost=.25,20,30,10
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image_array = np.asarray(image)[40:160,10:310,:]
        image_array=cv2.resize(image_array, (size_x,size_y)) # None,fx=.5,fy=.5,interpolation=cv2.INTER_CUBIC)
        transformed_image_array = image_array[None, :, :, :]
        try:
            out_steering_angle = float(model.predict(transformed_image_array, batch_size=1))
            #throttle = 0.5 + adjust_throttle_for_speed(speed,0,10,15) #+ adjust_throttle_for_steering(steering_angle, out_steering_angle)
            throttle = throt + adjust_throttle_for_speed(speed,mins,maxs,boost)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            out_steering_angle = 0
        print(throttle, out_steering_angle)
        send_control(out_steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    parser.add_argument(
        '--throttle', type=float, required=False, default=0.25,
        help='Default throttle.'
    )
    parser.add_argument(
        '--min', type=int, required=False, default=20,
        help='Minimum speed to target.'
    )
    parser.add_argument(
        '--max',  type=int, required=False, default=30,
        help='Maximum speed to target.'
    )
    parser.add_argument(
        '--boost', type=int, required=False, default=10,
        help='Throttle boost (or slowdown) when speed in outside specified range.'
    )
    args = parser.parse_args()

    if (args.throttle): throt=args.throttle
    if (args.min): mins=args.min
    if (args.max): maxs=args.max
    if (args.boost): boost=args.boost
    print('throt,mins,maxs,boost=',throt,mins,maxs,boost)
    model = load_model(args.model)
    _ign,size_y,size_x,_ignChannels = model.layers[0].input_shape
    print('image_size=(',size_x,size_y,')')

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
