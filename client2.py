import json
import time
import os
import socket

fp = open("example.json", "r")
if fp:
    memories = json.load(fp)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((socket.gethostname(), 49512))
    s.sendall(bytes(json.dumps(memories), encoding="utf-8"))

    recv_data = []
    recv_data_str = ""
    while True:
        data = s.recv(1024)
        if not data:
            break
        s.sendall(data)
        recv_data.append(data.decode('utf-8'))
        recv_data_str = ''.join(recv_data)
        try:
            stitched_json = json.loads(recv_data_str)
        except ValueError:
            continue
        break

    print(stitched_json)

