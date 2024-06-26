import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def analyze_color_distribution(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    # Convert image to HSV to isolate colors
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define thresholds for blue and red
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2

    # Calculate the percentage of blue and red pixels
    blue_percentage = np.sum(blue_mask) / (blue_mask.size * 255) * 100
    red_percentage = np.sum(red_mask) / (red_mask.size * 255) * 100

    return blue_percentage, red_percentage


def process_images_in_folder(folder_path):
    blue_percentages = []
    red_percentages = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        blue_percentage, red_percentage = analyze_color_distribution(image_path)
        if blue_percentage is not None and red_percentage is not None:
            blue_percentages.append(blue_percentage)
            red_percentages.append(red_percentage)

    return blue_percentages, red_percentages


def normalize_blues_to_two_colors(frame):
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define thresholds for the blue range in HSV
    lower_blue1 = np.array([90, 50, 50])
    upper_blue1 = np.array([130, 255, 255])

    # Create a mask for the blue areas
    mask = cv2.inRange(hsv, lower_blue1, upper_blue1)

    # Normalize blue areas to two distinct colors
    frame[mask != 0] = [255, 0, 0]  # Set blue areas to one distinct color (pure blue)
    frame[mask == 0] = [0, 0, 255]  # Set non-blue areas to another color (pure red)

    return frame


def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print("Error opening video file.")
        return

    normalised_dir = os.path.join(output_dir, 'normalised')
    original_dir = os.path.join(output_dir, 'original')

    if not os.path.exists(normalised_dir):
        os.makedirs(normalised_dir)
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)

    frame_index = 0
    while frame_index < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the center region of the frame
        height, width, _ = frame.shape
        centerX, centerY = width // 2, height // 2
        roi_size = int(
            min(width, height) * 2 / 3)  # Define the size of the ROI to be two-thirds of the smallest dimension
        center_frame = frame[centerY - roi_size // 2: centerY + roi_size // 2,
                       centerX - roi_size // 2: centerX + roi_size // 2]

        # Save the non-normalised frame
        original_output_path = os.path.join(original_dir, f'frame_{frame_index:02d}.png')
        cv2.imwrite(original_output_path, center_frame)

        # Normalize blues to two distinct colors
        normalized_frame = normalize_blues_to_two_colors(center_frame)

        # Save the normalised frame
        normalised_output_path = os.path.join(normalised_dir, f'frame_{frame_index:02d}.png')
        cv2.imwrite(normalised_output_path, normalized_frame)

        frame_index += 1

    cap.release()


output_dir = 'images'
# Delete all folders and files
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isdir(file_path):
        for image_file in os.listdir(file_path):
            os.remove(os.path.join(file_path, image_file))
        os.rmdir(file_path)
    else:
        os.remove(file_path)

video_path = 'video/final_video.mp4'
process_video(video_path, output_dir)

blue_percentages, red_percentages = process_images_in_folder('images/normalised')


def plot_color_variation(blue_percentages, red_percentages):
    plt.figure(figsize=(10, 5))
    plt.plot(blue_percentages, label='Blue Percentage')
    plt.plot(red_percentages, label='Red Percentage')
    plt.xlabel('Frame Index')
    plt.ylabel('Percentage')
    plt.title('Variation of Blue vs Red over Time')
    plt.legend()
    plt.show()


# Generate the graph
plot_color_variation(blue_percentages, red_percentages)
