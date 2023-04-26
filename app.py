import cv2
import mediapipe as mp
import boto3
from botocore.exceptions import NoCredentialsError

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
img_counter = 0

# Set up the S3 client
s3 = boto3.client('s3')

# Upload the PNG file to S3
bucket_name = 'face-recognition-3712'

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      k = cv2.waitKey(1)
      if k%256 == 32:
        # SPACE pressed
        file_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(file_name, image)
        print("{} written!".format(file_name))
        
        # Upload the PNG file to S3
        try:
          s3.upload_file(file_name, bucket_name, file_name)
          print("Upload Successful")
        except FileNotFoundError:
          print("The file was not found")
        except NoCredentialsError:
          print("Credentials not available")
        img_counter += 1

      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()