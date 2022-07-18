from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

def take_photo(filename1='Original.jpg', filename2='Canny.jpg', filename3='Sobel.jpg', filename4='Blur.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)

  # get photo data
  data = eval_js('takePhoto({})'.format(quality))
  # get OpenCV format image
  orImg = js_to_image(data)
  CannyImg = js_to_image(data)
  SobelImg = js_to_image(data)
  BlurImg =  js_to_image(data)
  # grayscale img
  gray = cv2.cvtColor(orImg, cv2.COLOR_RGB2GRAY)
  print(gray.shape)
  # get face bounding box coordinates using Haar Cascade
  faces = face_cascade.detectMultiScale(gray)
  # draw face bounding box on image
  for (x,y,w,h) in faces: #Canny Edge Detection
    #x=top, y=right, w=bottom, h=left
      CannyImg = cv2.rectangle(CannyImg,(x,y),(x+w,y+h),(255,0,0),2)
      CannyImg = cv2.cvtColor(CannyImg, cv2.COLOR_RGB2GRAY)
      face = CannyImg[y:y+h, x:x+w]
      #for Canny Image
      face = cv2.GaussianBlur(face,(23,23), 30)
      face = cv2.Canny(face, 20, 70)
      CannyImg[y:y+face.shape[0], x:x+face.shape[1]] = face

  for (x,y,w,h) in faces: #Sobel Edge Detection
      SobelImg = cv2.rectangle(SobelImg,(x,y),(x+w,y+h),(255,0,0),2)
      SobelImg = cv2.cvtColor(SobelImg, cv2.COLOR_RGB2GRAY)
      face = SobelImg[y:y+h, x:x+w]
      #for Sobel Image
      face = cv2.GaussianBlur(face,(23,23), 30)
      sobelx = cv2.Sobel(face, cv2.CV_64F,1,0,ksize=3)
      sobely = cv2.Sobel(face, cv2.CV_64F,0,1,ksize=3)
      face = sobelx + sobely
      SobelImg[y:y+face.shape[0], x:x+face.shape[1]] = face

  for (x,y,w,h) in faces: #Blur Image
      BlurImg = cv2.rectangle(BlurImg,(x,y),(x+w,y+h),(255,0,0),2)
      #BlurImg = cv2.cvtColor(BlurImg, cv2.COLOR_RGB2GRAY)
      face = BlurImg[y:y+h, x:x+w]
      #for blurry Image
      face = cv2.GaussianBlur(face,(23,23), 30)

      BlurImg[y:y+face.shape[0], x:x+face.shape[1]] = face

  # save image
  cv2.imwrite(filename1, orImg)
  cv2.imwrite(filename2, CannyImg)
  cv2.imwrite(filename3, SobelImg)
  cv2.imwrite(filename4, BlurImg)

  return filename1, filename2, filename3, filename4
  
  try:
  filename = take_photo('Original.jpg', 'Canny.jpg', 'Sobel.jpg', 'Blur.jpg')
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image('Original.jpg'))
  display(Image('Canny.jpg'))
  display(Image('Sobel.jpg'))
  display(Image('Blur.jpg'))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
