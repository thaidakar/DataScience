from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow import keras
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
   model = keras.models.load_model("trainedModel.h5")
   label = model.predict(batchedImage)
   return label