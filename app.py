from flask import Flask, render_template, Response, stream_with_context, request
import cv2
import argparse
from models import RULSTM
import torch
from torch import nn
from pretrainedmodels import bninception
from torchvision import transforms
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from os.path import basename
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from goprocam import GoProCamera, constants
from time import time
import socket
import random
import os
import io

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ip', type=str, default = 'rtmp://192.168.31.251',
                    help='IP address of the mona server, receiving video from GoPro')
parser.add_argument('--device', type=str, default = 'cpu',
                    help='cpu or cuda')
parser.add_argument('--goprocam', action='store_true',
                    help='use goprocam api if specified')
parser.add_argument('--run_model', action='store_true',
                    help='To perform inference. Otherwise - connection check mode')
parser.add_argument('--rtmp', action='store_true',
                    help='if camepa is connected with GoPro Quick app')
parser.add_argument('--folder', type=str, default=None,
                    help='to process pre-extracted frames from folder')
parser.add_argument('--fps', type=int, default=12,
                    help='Frames per second rate')
parser.add_argument('--finetuned', taction='store_true',
                    help='Use model fine-tuned on Homelab data')
args = parser.parse_args()
device = args.device

df = pd.read_csv('full_action_annots.csv')
id2act = dict(zip(df['action'].values, df['action_name'].values))

app = Flask(__name__, static_url_path='/static')

torch.set_grad_enabled(False)

transform = transforms.Compose([
    transforms.Resize([256, 454]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[[2,1,0],...]*255), #to BGR # ORIGINAL
    transforms.Normalize(mean=[104, 117, 128],
                         std=[1, 1, 1]),
])

selected_actions = {1821: 'wash hands', 1460: 'make coffee'
    1919: 'wash vegetables', 3532: 'cut vegetables',
    2334: 'agg sugar', 2626: 'add salt'
    2442: 'pour cola'}

if args.goprocam:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t=time()
    gpCam = GoProCamera.GoPro()
    gpCam.livestream("start")
    gpCam.video_settings(res='480p', fps=str(args.fps))
    EXTRACT_EVERY = int(args.fps/4)
    camera = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)
    print('Camera: ', camera)
if args.rtmp:
    print('Using rtmp stream from GoPro app')
    camera = cv2.VideoCapture(args.ip)
    EXTRACT_EVERY = int(args.fps/4)
if args.folder is not None:
    pass
    

def get_models():
    model = bninception(pretrained=None)
    state_dict = torch.load('TSN-rgb.pth.tar', map_location=torch.device('cpu'))['state_dict']
    state_dict = {k.replace('module.base_model.','') : v for k,v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.last_linear = nn.Identity()
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.to(device)
    model.eval()
    
    if args.finetuned:
        rulstm = RULSTM(num_class = 3805, feat_in=1024, hidden=1024)
        checkpoint = torch.load('RULSTM-anticipation_0.25_6_8_rgb_ft_best.pth.tar',
                                map_location=torch.device('cpu'))['state_dict']
        rulstm.load_state_dict(checkpoint)
    else:
        rulstm = RULSTM(num_class = 3806, feat_in=1024, hidden=1024)
        checkpoint = torch.load('RULSTM-anticipation_0.25_6_8_rgb_mt5r_best.pth.tar',
                                map_location=torch.device('cpu'))['state_dict']
        rulstm.load_state_dict(checkpoint)
    rulstm.to(device)
    rulstm.eval()
    
    return model, rulstm

def gen_frames():  # generate frame by frame from camera
    model, rulstm = get_models()
    print('Models are loaded')
    queue = []
    i = 0
    label = ''
    t=time()
    success, frame = camera.read()
    print('Success')
    while success:
        success, frame = camera.read()  # read the camera frame
        if not success:
            print('No Frame received')
            break
        else:
            pass
            i+=1
            if i%EXTRACT_EVERY == 0:
                if args.run_model:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8))
                    data = transform(img).unsqueeze(0).to(device)
                    feat = model(data).squeeze().detach().cpu().numpy()
                    queue.append(feat)
                    if len(queue)==15:
                        queue.pop(0)
                        feats = torch.Tensor(np.stack(queue)).unsqueeze(0)
                        if args.finetuned:
                            conf, y = torch.max(nn.Softmax()(rulstm(feats.to(device))[0,-1,:]), dim=0)
                            conf = conf.item()
                            y =y.item()
                            if y in selected_actions.keys() and conf>0.9:
                                label = str(selected_actions[y])
                            else:
                                label = ''
                        else:
                            label = id2act[int(torch.argmax(rulstm(feats)[0,-1,:]))]
                else:
                    label = id2act[random.randint(0,3804)]
            frame = cv2.putText(frame, label, org=(10,25), fontFace=0,
                    fontScale=1, color=(0, 0, 0), thickness=2)

            if time() - t >= 2.5:
                sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
                t=time()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def read_frames():
    model, rulstm = get_models()
    print('Models are loaded')
    arr_time = []
    i=0
    queue = []
    t=0
    font = ImageFont.truetype('US101.ttf', size=26)
    for i in range(len(os.listdir('out10/'))):
        if i%3 == 0:
            t = time()
            img = Image.open(args.folder+str(i)+'.jpg')
            data = transform(img).unsqueeze(0).to(device)
            feat = model(data).squeeze().detach().cpu().numpy()
            queue.append(feat)
            if len(queue)==15:
                queue.pop(0)
                feats = torch.Tensor(np.stack(queue)).unsqueeze(0)
                if args.finetuned:
                    conf, y = torch.max(nn.Softmax()(rulstm(feats.to(device))[0,-1,:]), dim=0)
                    conf = conf.item()
                    y =y.item()
                    if y in selected_actions.keys() and conf>0.9:
                        label = str(selected_actions[y])
                    else:
                        label = ''
                else:
                    y = id2act[int(torch.argmax(rulstm(feats)[0,-1,:]))]
                    label = selected_actions[y]
                t = time()
                if y in selected_actions.keys():
                    ImageDraw.Draw(img).text((0, 0),label,(0, 0, 0), font=font)
                else:
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                frame = buf.getvalue()
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                

@app.route('/video_feed')
def video_feed():
    if args.folder is not None:
        return Response(read_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)