import torch
from torch import nn
import numpy as np
import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
import boto3
import os


sequence_length = 32
device = 'cpu'
letter_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
pitch_max, pitch_min, duration_max, duration_min = (91, 53, 96, 0)
g6, f3 = pitch_max, pitch_min


#Downloading model
ACCESS_KEY = os.environ.get('ACCESS_KEY')
SECRET_ACCESS_KEY = os.environ.get('SECRET_ACCESS_KEY')
if not os.path.exists("./mdl.pth"):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY , aws_secret_access_key=SECRET_ACCESS_KEY)
    s3.download_file('pianoai','mdl.pth','./mdl.pth')

#Conversion function
def key_to_pitch(key):
    note, octave = tuple(key.split("/"))
    num_mod_12 = letter_notes.index(note)
    num_floordiv_12 = int(octave) + 1
    num = (12 * num_floordiv_12) + num_mod_12
    return num

#Inverse conversion (also edited to match desired format)
def pitch_to_key(num):
    letter = letter_notes[num % 12]
    octave = (num // 12) - 1
    return f"{letter}{octave}"



#Formatting
def model_input(x):
    duration_since_last = 0 #Since this is how the model is tracked
    returned_model_input = []
    for bar in x:
        notes = bar[1:-1].split("}")
        for note in notes:
            try:
                #Getting duration
                duration = note.split('duration":')[1].replace('"', '')
                n16 = {'q' : 4, '8' : 2, '16' : 1}[duration[0]]
                if 'r' in duration:
                    duration_since_last += n16
                else:
                    duration = n16
                    keys = note.split('keys":')[1].split(']')[0][1:].replace('"', '').split(',')
                    for idx, key in enumerate(keys):
                        #Formatting keys
                        key, oct = tuple(key.upper().split('/'))
                        oct = int(oct)
                        if len(key) > 1 and key[1] == 'B':
                            key_idx = letter_notes.index(key[0]) - 1
                            if key_idx == -1: key_idx = 0; oct -= 1;
                            key = letter_notes[key_idx]
                        pitch = key_to_pitch(f"{key}/{oct}")
                        
                        #Adding keys
                        returned_model_input.append([
                            (pitch - pitch_min) / (pitch_max - pitch_min), 
                            (duration_since_last - duration_min) / (duration_max - duration_min)
                        ])
                       
                        if idx == 0 and len(keys) > 1: duration_since_last = 0
                        elif idx == len(keys) - 1: duration_since_last = duration
                        
                                
                    
            except Exception as e:
                pass #Assume bad formatting from client app
    
    #Padding if needed
    if len(returned_model_input) < sequence_length:
        returned_model_input = [[0, 0] for _ in range(sequence_length-len(returned_model_input))] + returned_model_input
    else:
        returned_model_input = returned_model_input[-sequence_length:]
    return returned_model_input



#Model definition and generation function
class Regressor(nn.Module):
    def __init__(self, seqlen, hidden_dim, n_layers, drop, bidir, pooling_kernel_size):
        super().__init__()
        self.ff_projection = nn.Linear(2, hidden_dim)
        self.lstm_stack = nn.LSTM(hidden_dim, hidden_dim, n_layers, True, True, drop, bidir)
        #self.attn_stack = nn.MultiheadAttention(hidden_dim, n_layers, drop)
        self.pooling = nn.MaxPool2d(pooling_kernel_size)
        self.ff1 = nn.Linear(int(seqlen * hidden_dim / ((pooling_kernel_size)**2)), 512)
        if bidir:
            self.ff1 = nn.Linear(2 * int(seqlen * hidden_dim / ((pooling_kernel_size)**2)), 512)
        self.ff2 = nn.Linear(512, 64)
        self.out = nn.Linear(64, 2)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.tensor(x).type(torch.float32)
        x = self.ff_projection(x)
        x, _ = self.lstm_stack(x)
        #x, _ = self.attn_stack(x, x, x)
        x = self.pooling(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.ff1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.ff2(x)
        x = self.relu(x)
        x = self.out(x)
        return x


model = Regressor(
    seqlen = sequence_length,
    hidden_dim = 256,
    n_layers = 4,
    drop = 0.1,
    bidir = True,
    pooling_kernel_size = 2
).to(device)


#Reopening and loading model
with open('./mdl.pth', 'rb') as f:
    state_dict = torch.load(f, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

#Generation function
def gen_reg(x, length):
    init_len = len(x)
    while len(x) < init_len + length:
        pred = model(torch.tensor([x[-sequence_length:]]).to(device)).tolist()[0]
        x = x + [pred]

    #Reverse standardizing
    gend = []
    for pred in x[-length:]:
        p, d = pred[0], pred[1]
        if d < 0: d = 0
        next_entry = [
            int((p * (pitch_max - pitch_min)) + pitch_min),
            int((d * (duration_max - duration_min)) + duration_min)
        ]
        
    
        #Octave correction
        next_entry[0] = int(next_entry[0])
        while next_entry[0] < f3:
            next_entry[0] += 12
        while next_entry[0] > g6:
            next_entry[0] -= 12
        
        gend.append(next_entry)
    return gend


#Function to generate from request and return a format needed
def generate_from_notes(notes):

    #Preprocessing and model output
    mdl_input = model_input(notes)
    mdl_output = gen_reg(mdl_input, 16)


    #Postprocessing and return 
    generated_list = []
    keys_to_add = []
    for idx, el in enumerate(mdl_output):
        keys_to_add.append(pitch_to_key(el[0]))
        if el[1] != 0:
            #Getting how many rests to add if the duration requires it, and adding all of it
            complete_duration = el[1]
            possibilities = [1, 2, 4]
            poss_keys = {4 : '4', 2 : '8', 1 : '16'} #To get it in proper vexflow format
            while complete_duration not in possibilities and complete_duration > 0:
                for p in possibilities:
                    if complete_duration > p:
                        generated_list.append({'note' : "Bb4", 'duration' : f'{poss_keys[p]}rn'})
                        complete_duration -= p
                        break

            if complete_duration > 0: #Should be, but just to avoid errors
                generated_list.append({'note' : np.unique(','.join(keys_to_add)).tolist(), 'duration' : f'{poss_keys[complete_duration]}n'})
            
            keys_to_add = []

    ret = {}
    for idx, i in enumerate(generated_list):
        ret[str(idx)] = i

    return json.dumps(ret)



#Creating and using the flask app
app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods = ['GET']) #For debugging
@cross_origin()
def get():
    return "PianoAI"


@app.route("/gen", methods = ['POST'])
@cross_origin()
def api_handler():
    try:
        data = request.data.decode('utf-8')
        generated = generate_from_notes(data[0])
        return generated
    except Exception as e:
        return json.dumps({"error" : e})

    
