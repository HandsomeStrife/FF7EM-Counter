import sys

import cv2
import numpy as np
import os
import re
import subprocess
import time
from obswebsocket import obsws, requests
from tqdm import tqdm


def list_video_devices():
    """
    List available video devices using FFmpeg.
    :return:
    """
    cmd = "ffmpeg -f avfoundation -list_devices true -i \"\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stderr.splitlines()
    video_devices = []
    device_pattern = re.compile(r"\[\d+\] .+")

    for line in lines:
        if "AVFoundation video devices:" in line:
            continue  # Skip the header line
        match = device_pattern.search(line)
        if match:
            device_info = match.group(0)
            index = device_info.split(']')[0][1:].strip()  # Extract index between [ and ]
            name = device_info.split(']')[1].strip()  # Extract name after ]
            if index.isdigit():
                video_devices.append((index, name))
    return video_devices


def connect_to_obs(password):
    """
    Connect to the users OBS instance using the OBS WebSocket plugin.
    :return:
    """
    host = "localhost"
    port = 4455
    client = obsws(host, port, password)
    try:
        client.connect()
        print("Connected to OBS.")
    except Exception as e:
        print("Failed to connect to OBS:", e)
        return None
    return client


def monitor_and_record(cap, template_path, output_path):
    template = cv2.imread(template_path, 0)  # Load template in grayscale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False
    matched_region = None
    matched_scale = None

    # Get the frame rate of the source video
    fps = 30.0

    print(f"Source video FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not recording:
            match = multi_scale_template_matching(frame, template)
            if match is not None:
                print("Template matched, starting recording...")
                output_video_path = os.path.join(output_path, 'recorded_video.mp4')
                matched_region = match  # Store the coordinates of the matched region
                startX, startY, endX, endY, matched_scale = matched_region

                # Adjust the region to use the width of the template and double its height
                region_width = endX - startX
                region_height = endY - startY
                endY = startY + region_height

                # Ensure the region dimensions are within frame bounds
                endY = min(endY, frame.shape[0])

                out = cv2.VideoWriter(output_video_path, fourcc, fps, (region_width, endY - startY))
                recording = True

        if recording:
            if matched_region is not None:
                startX, startY, endX, endY, matched_scale = matched_region
                # Adjust the region to use the width of the template and double its height
                region_width = endX - startX
                region_height = endY - startY
                endY = startY + region_height

                # Ensure the region dimensions are within frame bounds
                endY = min(endY, frame.shape[0])

                region = frame[startY:endY, startX:endX]
                mean_brightness = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
                if mean_brightness < 50:  # Adjust this threshold based on your video content
                    print("Screen is fading to black, stopping recording...")
                    break
                out.write(region)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if recording:
        out.release()


def multi_scale_template_matching(frame, template, fixed_scale=None, threshold=0.8, debug=False):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (frameH, frameW) = gray_frame.shape[:2]
    found = None
    scales = [fixed_scale] if fixed_scale else np.linspace(1.0, 0.2, num=20)

    # Loop over the scales of the template
    for scale in scales:
        resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
        (tH, tW) = resized_template.shape[:2]

        if tH > frameH or tW > frameW:
            continue

        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

        if debug:
            print(f"Scale: {scale:.2f}, MaxVal: {maxVal}")

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, scale)

    if found is None:
        return None

    (maxVal, maxLoc, scale) = found
    if maxVal < threshold:  # Adjust threshold based on findings
        return None

    (startX, startY) = maxLoc
    (endX, endY) = (startX + int(template.shape[1] * scale), startY + int(template.shape[0] * scale))

    if debug:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow('Match', frame)
        cv2.waitKey(0)  # Wait for a key press to continue

    return startX, startY, endX, endY, scale


def crop_recorded_video(recorded_video_path, templates_paths, output_cropped_path, save_failed_path, margin=50):
    print("Cropping the recorded video...")
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {recorded_video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = None
    out = None
    last_scale = None
    missed_frames = 0

    if not os.path.exists(save_failed_path):
        os.makedirs(save_failed_path)

    templates = [cv2.imread(path, 0) for path in templates_paths]
    fallback_template = cv2.imread(templates_paths[0], 0)
    fallback_region = fallback_template[int(2 * fallback_template.shape[0] / 3):, :]

    pbar = tqdm(total=total_frames, desc="Cropping frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            pbar.close()
            break

        match_found = False
        for template in templates:
            match = multi_scale_template_matching(frame, template, fixed_scale=last_scale, threshold=0.9, debug=False)
            if match:
                match_found = True
                startX, startY, endX, endY, scale = match
                last_scale = scale
                midY = startY + (endY - startY) // 2
                cropped_region = frame[midY:endY, startX:endX]

                if out is None and frame_size is None:
                    frame_height = endY - midY
                    frame_size = (endX - startX, frame_height)
                    out = cv2.VideoWriter(output_cropped_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

                if out is not None:
                    out.write(cropped_region)
                break

        if not match_found:
            match = multi_scale_template_matching(frame, fallback_region, fixed_scale=last_scale, threshold=0.9,
                                                  debug=False)
            if match:
                startX, startY, endX, endY, scale = match
                last_scale = scale
                midY = startY + (endY - startY) // 2
                cropped_region = frame[midY:endY, startX:endX]

                if out is None:
                    frame_height = endY - midY
                    frame_size = (endX - startX, frame_height)
                    out = cv2.VideoWriter(output_cropped_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

                if out is not None:
                    out.write(cropped_region)
            else:
                missed_frames += 1
                if last_scale:
                    # Save the last attempted region with margin if no match was found
                    sx = max(startX - margin, 0)
                    sy = max(startY - margin, 0)
                    ex = min(endX + margin, frame.shape[1])
                    ey = min(endY + margin, frame.shape[0])
                    failed_region_with_margin = frame[sy:ey, sx:ex]
                    failed_img_path = os.path.join(save_failed_path, f"failed_frame_{pbar.n}.png")
                    cv2.imwrite(failed_img_path, failed_region_with_margin)

        pbar.update(1)

    cap.release()
    if out:
        out.release()

    print(f"Cropping completed: Processed {pbar.n} frames of {total_frames}")
    print(f"Missed frames: {missed_frames}")


def analyze_center_color_changes(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video

    colors_over_time = []  # Store the average color per frame in the center region
    significant_changes = []  # Store times and magnitude of significant color changes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the region of interest (ROI) as the center of the frame
        height, width, _ = frame.shape
        centerX, centerY = width // 2, height // 2
        roi_size = min(width, height) // 4  # Define the size of the ROI to be a quarter of the smallest dimension
        roi = frame[centerY - roi_size // 2 : centerY + roi_size // 2,
                    centerX - roi_size // 2 : centerX + roi_size // 2]

        # Convert ROI to HSV to better analyze color changes
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        average_color = cv2.mean(hsv_roi)
        colors_over_time.append(average_color[:3])  # Append HSV averages, excluding the alpha channel if present

    cap.release()

    # Analyze changes between consecutive frames
    for i in range(1, len(colors_over_time)):
        prev_color = np.array(colors_over_time[i - 1])
        curr_color = np.array(colors_over_time[i])

        # Calculate the Euclidean distance between color vectors in HSV space
        color_change = np.linalg.norm(curr_color - prev_color)
        if color_change > 5:  # Threshold for detecting a significant change
            time_of_change = i / fps  # Calculate the time of change based on frame index and FPS
            significant_changes.append(time_of_change)

    # Calculate intervals between detected glows
    intervals = [round(significant_changes[i] - significant_changes[i - 1], 2) for i in range(1, len(significant_changes))]
    # Filter out intervals less than 0.10 seconds
    filtered_intervals = [interval for interval in intervals if interval >= 0.10]
    return filtered_intervals


def main():
    # # Check if we have a password file
    # if not os.path.exists("obs_password.txt"):
    #     # Ask for the password and save it to a file
    #     password = input("Enter the OBS WebSocket password: ")
    #     with open("obs_password.txt", "w") as f:
    #         f.write(password)
    #
    # # Read the password from the file
    # with open("obs_password.txt", "r") as f:
    #     password = f.read().strip()
    #
    # # User chooses the input source
    # choice = input("Enter 'OBS' to use OBS Virtual Camera or 'file' to use a video file: ")
    #
    # if choice.lower() == 'obs':
    #     client = connect_to_obs(password)
    #     if not client:
    #         return
    #
    #     # Check virtual camera status
    #     virtual_cam_status = client.call(requests.GetVirtualCamStatus())
    #     if not virtual_cam_status.getOutputActive():
    #         print("Virtual Camera is not active, starting it...")
    #         client.call(requests.StartVirtualCam())
    #         time.sleep(2)  # Wait a moment for the virtual camera to start
    #
    #     # List available video devices
    #     print("Listing available video devices...")
    #     video_devices = list_video_devices()
    #
    #     # Find OBS Virtual Camera
    #     new_camera_index = None
    #     for index, name in video_devices:
    #         if name == "OBS Virtual Camera":
    #             new_camera_index = int(index)
    #             break
    #
    #     if new_camera_index is None:
    #         print("Failed to identify the OBS Virtual Camera.")
    #         client.disconnect()
    #         return
    #
    #     print(f"Using OBS Virtual Camera with index: {new_camera_index}")
    #     cap = cv2.VideoCapture(new_camera_index)
    #
    # elif choice.lower() == 'file':
    #     video_file_path = input("Enter the path to the video file: ")
    #     cap = cv2.VideoCapture(video_file_path)
    #     if not cap.isOpened():
    #         print(f"Failed to open video file: {video_file_path}")
    #         return
    #
    # else:
    #     print("Invalid input. Exiting...")
    #     return
    #
    # template_path = 'reference/aerith.png'  # Path to the template image
    # output_path = 'video/'  # Path to save the output video
    #
    # # Delete any existing video
    # if os.path.exists(output_path):
    #     for file in os.listdir(output_path):
    #         file_path = os.path.join(output_path, file)
    #         if os.path.isfile(file_path) and file != ".gitignore":
    #             os.remove(file_path)
    # else:
    #     os.makedirs(output_path)
    #
    # monitor_and_record(cap, template_path, output_path)
    #
    # cap.release()
    # cv2.destroyAllWindows()
    # if choice.lower() == 'obs':
    #     client.disconnect()
    #
    # # Crop the recorded video
    # recorded_video_path = os.path.join(output_path, 'recorded_video.mp4')
    #
    # template_paths = ['reference/1.png', 'reference/2.png', 'reference/3.png']
    # output_cropped_path = os.path.join(output_path, 'cropped_video.mp4')
    #
    # save_failed_path = 'images'
    # crop_recorded_video(recorded_video_path, template_paths, output_cropped_path, save_failed_path)

    output_cropped_path = 'video/cropped_video.mp4'
    intervals = analyze_center_color_changes(output_cropped_path)
    intervals.reverse()
    print("Glow intervals in seconds:", intervals)

    # # Detect significant changes in blue light intensity
    # significant_times = detect_blue_light_changes(output_cropped_path, threshold=1)
    #
    # # Calculate intervals between significant changes
    # intervals = calculate_intervals(significant_times)
    #
    # # Reverse the intervals
    # intervals.reverse()
    #
    # # Remove any from the start that are less than 0.1 seconds
    # intervals = [interval for interval in intervals if interval > 0.1]
    #
    # compare = ['1.49', '2.13', '1.44', '1.48', '2.75', '2.16', '1.43', '1.46', '2.13']
    # compare_two = [1, 2, 1, 1, 3, 2, 1, 1, 2]
    #
    # for i, interval in enumerate(intervals):
    #     if i < len(compare):
    #         print(f"Interval {i + 1}: {interval}s | {compare[i]} | {compare_two[i]}")


if __name__ == "__main__":
    main()
