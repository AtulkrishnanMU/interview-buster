import cv2

# Set the video source (0 for default camera)
video_source = 0

def live_save():
    # Open the video capture
    cap = cv2.VideoCapture(0)

    # Get the default video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('live_rec.mp4', fourcc, 20.0, (width, height))

    while True:
        # Read a frame from the video source
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Display the frame (optional)
        cv2.imshow('Frame', frame)

        # Write the frame to the output video file
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    '''
    if record_button:

        # Add a message for recording in progress
        recording_text = st.text("Recording in progress... Click the button again to stop recording.")

        live_save()

        recording_text.empty()  # Clear the recording in progress message

        # Now, you can use the recorded video path as file_path
        file_path = r"E:\project demo\live_rec.mp4"
        uploaded = 1
    '''

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
