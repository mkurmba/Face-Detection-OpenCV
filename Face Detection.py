import cv2

# Load the face cascade classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

def detectFaceBox(frame):
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #this line uses the fase_classifier object from openCV library to detect faces
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # get the values returned from detectMultiScale function.
    # x and y are the coordinates of the top-left corner of the face rectangle,
    # w is the width of the rectangle, and h is the height of the rectangle.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)  #color of the box is set to red


while True:
    # Read frames from the video capture
    success, frame = video_capture.read()

    if not success:
        break  # Terminate the loop if the frame is not read successfully

    #input the frame in the detectFaceBox to detect all the faces
    detectFaceBox(frame)

    # Display the frame with face detection
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and destroy any remaining windows
video_capture.release()
cv2.destroyAllWindows()
