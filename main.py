import cv2
import numpy as np
import os
import re
import subprocess
import time
from obswebsocket import obsws, requests


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


def multi_scale_template_matching(frame, template, debug=False):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (frameH, frameW) = gray_frame.shape[:2]
    found = None

    # Loop over the scales of the template
    for scale in np.linspace(1.0, 0.2, num=20):
        resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
        (tH, tW) = resized_template.shape[:2]

        if tH > frameH or tW > frameW:
            continue

        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

        # if debug:
        #     print(f"Scale: {scale:.2f}, MaxVal: {maxVal}")
        #     cv2.imshow('Resized Template', resized_template)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, scale)

    if found is None:
        return None

    (maxVal, maxLoc, scale) = found
    if maxVal < 0.8:  # Adjust threshold based on findings
        return None

    (startX, startY) = maxLoc
    (endX, endY) = (startX + int(template.shape[1] * scale), startY + int(template.shape[0] * scale))

    if debug:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow('Match', frame)
        cv2.waitKey(0)  # Wait for a key press to continue

    return startX, startY, endX, endY, scale


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


def crop_recorded_video(recorded_video_path, section_template_path, output_cropped_path):
    section_template = cv2.imread(section_template_path, 0)  # Load the section template in grayscale
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {recorded_video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        match = multi_scale_template_matching(frame, section_template, debug=False)
        if match is not None:
            startX, startY, endX, endY, scale = match
            cropped_region = frame[startY:endY, startX:endX]

            if out is None:
                # Initialize the VideoWriter for the cropped video
                out = cv2.VideoWriter(output_cropped_path, fourcc, fps, (endX - startX, endY - startY))

            out.write(cropped_region)

    cap.release()
    if out is not None:
        out.release()


def main():
    # Check if we have a password file
    if not os.path.exists("obs_password.txt"):
        # Ask for the password and save it to a file
        password = input("Enter the OBS WebSocket password: ")
        with open("obs_password.txt", "w") as f:
            f.write(password)

    # Read the password from the file
    with open("obs_password.txt", "r") as f:
        password = f.read().strip()

    client = connect_to_obs(password)
    if not client:
        return

    # Check virtual camera status
    virtual_cam_status = client.call(requests.GetVirtualCamStatus())
    if not virtual_cam_status.getOutputActive():
        print("Virtual Camera is not active, starting it...")
        client.call(requests.StartVirtualCam())
        time.sleep(2)  # Wait a moment for the virtual camera to start

    # List available video devices
    print("Listing available video devices...")
    video_devices = list_video_devices()

    # Find OBS Virtual Camera
    new_camera_index = None
    for index, name in video_devices:
        if name == "OBS Virtual Camera":
            new_camera_index = int(index)
            break

    if new_camera_index is None:
        print("Failed to identify the OBS Virtual Camera.")
        client.disconnect()
        return

    print(f"Using OBS Virtual Camera with index: {new_camera_index}")
    cap = cv2.VideoCapture(new_camera_index)

    template_path = 'reference/aerith.png'  # Path to the template image
    output_path = 'video/'  # Path to save the output video

    # Delete any existing video
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_path)

    monitor_and_record(cap, template_path, output_path)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

    # Crop the recorded video
    recorded_video_path = os.path.join(output_path, 'recorded_video.mp4')
    section_template_path = 'reference/section.png'  # Path to the section template image
    output_cropped_path = os.path.join(output_path, 'cropped_video.mp4')

    crop_recorded_video(recorded_video_path, section_template_path, output_cropped_path)


if __name__ == "__main__":
    main()
