from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
from keras.models import load_model
import cv2
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import Response

limiter = Limiter(key_func=get_remote_address) # Gets the user's address
app = FastAPI()
app.state.limiter = limiter # Define's the api's limiter object
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # Adds the exception to the api

@app.get("/base/home")
@limiter.limit("15/minute")
async def classify(request:Request, response:Response):
    return {"api_info": "Sean - Brady's CS280 Image Classifier"}


@app.post("/classify/")
@limiter.limit("5/minute")
async def classify(request:Request, response:Response, file: UploadFile = File(...)):
    fileContent = await file.read() # Reads file information from request as a stream of bytes
    batchedImage = convertFileToBatchedImage(fileContent)
    label = predictImage(batchedImage)
    return {"classification": f"{label}"} # Returns the filename as a response.


def convertFileToBatchedImage(fileContent):
  image = Image.open(BytesIO(fileContent)) # Converts file to a python image from a stream of bytes.
  imageArray = np.asarray(image) #converts image to numpy array
  smallImage = cv2.resize(imageArray, dsize=(32, 32), interpolation=cv2.INTER_CUBIC) # Resizes the image
  batchedImage = np.expand_dims(smallImage, axis=0)
  return batchedImage

def predictImage(batchedImage):
  model = load_model("trainedModel.h5")
  # predict the class
  result = model.predict(batchedImage)
  # convert the probabilities to class labels
  label = np.argmax(result, axis=1)[0]

  LABELS = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'] 

  return LABELS[label]