# gopro_stream_demo_aa
Live demo on egocentric action anticipation from gopro camera
![goprostreaming](https://user-images.githubusercontent.com/17033647/201895296-f50c45da-ae2e-49bb-a993-c1c583a813bd.png)

### Requirements
This repo relies on `gopro-py-api` (https://github.com/KonradIT/gopro-py-api)

Compatibility:

- HERO3
- HERO3+
- HERO4 (including HERO Session)
- HERO+
- HERO5 (including HERO5 Session)
- HERO6
- Fusion 1
- HERO7 (Black)
- HERO8 Black
- MAX
- HERO9 Black
- HERO10 Black

### Setting up

The demonstration can work in 3 settings:

1. Laptop connected to camera over Wi-Fi:
`python main.py --goprocam`
With this setting you should first turn-on Wi-Fi on the camera, and connect the laptop to this network before running the script. This is fastest setting for live streaming, however, the glitching can occur, and frame quality may not be the best.

2. RTMP connection via GoPro app:
`python main.py --rtmp --ip 1.1.1.1`
In this setting the frames are first streamed to the mobile GoPro app, and from this app are then sent to the RTMP server pre-configured on laptop. The frame resolution is high, but the transition speed is low, big broadcast delay.

3. Demo with pre-extracted frames stored in a folder:
`python main.py --folder path/to/frames`
