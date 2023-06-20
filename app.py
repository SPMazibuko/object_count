import os
import io
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
import cv2
import argparse
import datetime
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response,jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from waitress import serve
from typing import List

model = YOLO('best.pt')

app = Flask(__name__)

static_folder = 'static'
new_folder = 'uploads'
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/detect_shelves', methods=['POST'])
def detect_shelves():
    image_file = request.files['file']
    if image_file:
        
        uploads_folder = os.path.join(static_folder, new_folder)
        if not os.path.exists(uploads_folder):
            os.makedirs(uploads_folder)
        
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(uploads_folder, image_file.filename)
        image_file.save(image_path)

        detector = ShelfDetector('best.pt')
        result = detector.detect_shelves(image_path)
        shelf_info = [{'shelf_number': i+1, 'product_count': size} for i, size in enumerate(result)]
        return render_template('result.html', shelves=shelf_info)

    return 'No image file uploaded.'
    
class ShelfDetector:
    def __init__(self, model_path: str, confidence: float = 0.45):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_shelves(self, image_path: str) -> List[int]:
        result = self.model.predict(source=image_path, conf=self.confidence, save=False)
        arrxy = result[0].boxes.xyxy
        coordinates = np.array(arrxy)

        x_coords = (coordinates[:, 0] + coordinates[:, 2]) / 2
        y_coords = (coordinates[:, 1] + coordinates[:, 3]) / 2
        midpoints = np.column_stack((x_coords, y_coords))

        sorted_midpoints = midpoints[midpoints[:,1].argsort()]
        rounded_n_sorted_arr = np.round(sorted_midpoints).astype(int)

        group_sizes = []
        objects = 0
        for i in range(1, len(rounded_n_sorted_arr)):
            if rounded_n_sorted_arr[i][1] - rounded_n_sorted_arr[i-1][1] > 130:
                group_sizes.append(objects + 1)
                objects = 0
            else:
                objects += 1

        group_sizes.append(objects + 1)
        return group_sizes
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to the YOLO model')
    args = parser.parse_args()
    
    detector = ShelfDetector(args.model_path)
    app.run(debug=True, host='0.0.0.0', port=5000)
