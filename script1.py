import cv2

def take_screenshot():
    # Open the camera
    camera = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Unable to open the camera.")
        return
    
    # Read a frame from the camera
    ret, frame = camera.read()
    
    # Check if the frame is read successfully
    if not ret:
        print("Unable to read frame from the camera.")
        camera.release()
        return
    
    # Save the frame as a screenshot
    cv2.imwrite("screenshot.png", frame)
    
    # Release the camera
    camera.release()
    
    print("Screenshot taken successfully.")

# Call the function to take a screenshot
take_screenshot()

