import requests
import cv2
from os import environ
from dotenv import load_dotenv

load_dotenv()

# function for image classification, huggingface inference api
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": f"Bearer {environ['TOKEN']}"}

def imageClassify(img):
    response = requests.post(API_URL, headers=headers, data=img.getvalue())
    return response.json()

# function for image captioning, huggingface inference api
API_URL_2 = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

def imageCaption(img):
    response = requests.post(API_URL_2, headers=headers, data=img.getvalue())
    return response.json()

# function to convert to grayscale, opencv
def convToGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# function to detect edges using Canny edge detection algorithm, opencv
def detectEdges(img):
    return cv2.Canny(img, 100, 200)

# function to detect faces using Cascade classifier, opencv
def detectFaces(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        center = (x + w//2, y + h//2)
        cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 2)
    return img