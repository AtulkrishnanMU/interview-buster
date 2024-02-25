import cv2
import os

def video_to_images(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through each frame and save it as an image
    for frame_number in range(frame_count):
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image
        image_name = f"frame_{frame_number:04d}.png"
        image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(image_path, frame)

        # Print progress
        print(f"Saving frame {frame_number}/{frame_count}")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Replace 'input_video.mp4' with the path to your input video file
    input_video_path = r'E:\project demo\output_video4.mp4'

    # Replace 'output_frames' with the desired output folder for image sequences
    output_folder = r'E:\website files\pose\sequence2'

    # Call the function to convert video to image sequences
    video_to_images(input_video_path, output_folder)

    print("Conversion complete.")
