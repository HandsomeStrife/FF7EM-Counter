# Waterfall Counter

A proof-of-concept for a waterfall counter, to help with the elevator manipulation in FF7.

## Requirements

**Python 3.12** - 
It might work on lower versions of Python, but I've only tried it on 3.12

**OBS** - You will need to have OBS installed and running, preferably the latest version.

**FFMpeg** - You can check if you have FFMPEG installed by running `ffmpeg -version` in your terminal.

## Setup
You will need to have OBS running with the websocket server running. 
To do this, go to `Tools` -> `WebSockets Server Settings` and enable the websocket server. 
You should enable authentication (for your safety), copy the password, and ensure its listening on port 4455.

The software will make use of your virtual camera on OBS, so you'll need to ensure you're not already using it for something else.

## Running the script
To run the script, you can simply run `python main.py` in your terminal, and follow the instructions.
You can run the script at any time, but it's preferable to run it before you arrive at Aerith's house for the first time.
It can stay active from the start of your run if you like, but once it's run once you would need to reset it.

Once you arrive at Aerith's house, it will detect your arrival and start recording the screen. When the screen fades to black,
it will stop recording and begin to crop the image to just the relevant section of the waterfall.