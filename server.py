import socket
import threading
import argparse
import json
import hashlib

import traceback
from pathlib import Path
from datetime import datetime
from hashlib import sha256

import gc
import torch

from getconfig import config, setting_info
from utils import *
from gpt2generator import GPT2Generator
from interface import instructions

parser = argparse.ArgumentParser()

parser.add_argument(
    "--host",
    default=socket.gethostname(),
    type=str,
    required=False,
    help="Host for which the server connects to.\n",
)

parser.add_argument(
    "--port",
    default=49512,
    type=int,
    required=False,
    help="Port for the server to listen to\n",
)

parser.add_argument(
    "--password",
    default="",
    type=str,
    required=False,
    help="Password for people to connect to the server\n",
)

parser.add_argument(
    "--name",
    default="DungeonServer",
    type=str,
    required=False,
    help="Name of the server that people see when they try to connect to it.\n",
)

args = parser.parse_args()

def encrypt_string(hash_string):
    sha_signature = \
        hashlib.sha256(hash_string.encode()).hexdigest()
    return sha_signature

def get_generator():
    output(
        "\nInitializing AI Engine! (This might take a few minutes)",
        "loading-message", end="\n\n"
    )
    models = [x for x in Path('models').iterdir() if x.is_dir()]
    generator = None
    failed_env_load = False
    while True:
        try:
            transformers_pretrained = os.environ.get("TRANSFORMERS_PRETRAINED_MODEL", False)
            if transformers_pretrained and not failed_env_load:
                # Keep it as a string, so that transformers library will load the generic model
                model = transformers_pretrained
                assert isinstance(model, str)
            else:
                # Convert to path, so that transformers library will load the model from our folder
                if not models:
                    raise FileNotFoundError(
                        'There are no models in the models directory! You must download a pytorch compatible model!')
                if os.environ.get("MODEL_FOLDER", False) and not failed_env_load:
                    model = Path("models/" + os.environ.get("MODEL_FOLDER", False))
                elif len(models) > 1:
                    output("You have multiple models in your models folder. Please select one to load:", 'message')
                    list_items([m.name for m in models] + ["(Exit)"], "menu")
                    model_selection = input_number(len(models))
                    if model_selection == len(models):
                        output("Exiting. ", "message")
                        exit(0)
                    else:
                        model = models[model_selection]
                else:
                    model = models[0]
                    logger.info("Using model: " + str(model))
                assert isinstance(model, Path)
            generator = GPT2Generator(
                model_path=model,
                generate_num=settings.getint("generate-num"),
                temperature=settings.getfloat("temp"),
                top_k=settings.getint("top-k"),
                top_p=settings.getfloat("top-p"),
                repetition_penalty=settings.getfloat("rep-pen"),
            )
            break
        except OSError:
            if len(models) == 0:
                output("You do not seem to have any models installed.", "error")
                output("Place a model in the 'models' subfolder and press enter", "error")
                input("")
                # Scan for models again
                models = [x for x in Path('models').iterdir() if x.is_dir()]
            else:
                failed_env_load = True
                output("Model could not be loaded. Please try another model. ", "error")
            continue
        except KeyboardInterrupt:
            output("Model load cancelled. ", "error")
            exit(0)
    return generator

back_log = []
back_log_event = threading.Event()
result = ""

def new_client(s, csock, addr, event):
    recv_data = []
    recv_data_str = ""
    while True:
        data = csock.recv(1024)
        if not data:
            break
        recv_data.append(data.decode('utf-8'))
        recv_data_str = ''.join(recv_data)
        try:
            stitched_json = json.loads(recv_data_str)
        except ValueError:
            continue
        break
    
    if args.password != str(""):
        if stitched_json["pass"] != encrypt_string(args.password):
            print(str(addr) + ' -- Disconnected: Invalid Password')
            csock.close()
            return
    
    # Enter object into backlog.
    back_log.append([stitched_json, event])
    # Wait for stuff to process from backlog.
    back_log_event.set()
    event.wait()

    sendpacket = {
        "result": result
    }
    
    print(str(addr) + " -- Serviced successfully")

    csock.sendall(bytes(json.dumps(sendpacket), encoding="utf-8"))
    csock.close()
    print(str(addr) + " -- Connection closed")

def connection_listener(s):
    print('Server started...')
    s.listen()

    while True:
        c, addr = s.accept()
        print(str(addr) + " -- Connection opened")
        client_handler = threading.Thread(target=new_client, args=(s, c, addr, threading.Event()))
        client_handler.start()


# Start the GPT-2 Generator

generator = get_generator()

# Start the connection.

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((args.host, args.port))
conn_listener = threading.Thread(target=connection_listener, args=(s,))
conn_listener.start()
print('Started conn_listener')

try:
    while True:
        # process queue
        back_log_event.wait()

        for i in back_log:
            result = generator.generate(context=i[0]["context"], prompt=i[0]["prompt"], temperature=i[0]["temp"], top_p=i[0]["top_p"], top_k=20, repetition_penalty=i[0]["rep_pen"])
            i[1].set()

        back_log_event.clear()
except KeyboardInterrupt:
    conn_listener.join()
    s.close()
    print('Server closed...')
