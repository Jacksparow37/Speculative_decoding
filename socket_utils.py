# socket_utils.py
import socket
import pickle
import os
import torch

# --- Server Process (on cuda:0) ---
def server_process(gpu_index, output_queue, host='localhost', port=5000):
    """Server process that listens for messages from clients and sends acknowledgments."""
    # Set the GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = torch.device("cuda:0")  # After setting CUDA_VISIBLE_DEVICES, cuda:0 maps to the correct GPU
    print(f"Server running on {device}")

    # Set up the server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(2)  # Listen for 2 clients
    print(f"Server started, listening on {host}:{port}")

    # Accept connections from clients
    clients = []
    for _ in range(2):  # Expecting 2 clients
        client_socket, addr = server_socket.accept()
        clients.append((client_socket, addr))
        print(f"Server accepted connection from {addr}")

    # Process messages in FIFO order
    received_messages = []
    active_clients = len(clients)
    while active_clients > 0:
        for client_socket, addr in clients:
            try:
                # Receive the length of the incoming data
                data_length_bytes = client_socket.recv(4)
                if not data_length_bytes:
                    active_clients -= 1
                    continue
                data_length = int.from_bytes(data_length_bytes, byteorder='big')

                # Receive the data
                data = b""
                while len(data) < data_length:
                    packet = client_socket.recv(data_length - len(data))
                    if not packet:
                        break
                    data += packet

                if not data:
                    active_clients -= 1
                    continue

                # Deserialize the data
                message = pickle.loads(data)
                client_id = message["client_id"]
                text = message["text"]
                received_messages.append((client_id, text))
                print(f"Server received message from Client {client_id}: {text}")

                # Send acknowledgment back to the client
                client_socket.sendall("ACK".encode())

            except (ConnectionResetError, BrokenPipeError):
                active_clients -= 1
                print(f"Connection closed by Client {client_id}")
                continue

    # Close all client sockets and the server socket
    for client_socket, _ in clients:
        client_socket.close()
    server_socket.close()
    print("Server shut down")

    # Send the received messages back to the main process via the queue
    output_queue.put(received_messages)

# --- Client Process (on cuda:1 or cuda:2) ---
def client_process(gpu_index, client_id, messages, host='localhost', port=5000):
    """Client process that sends messages to the server and waits for acknowledgments."""
    # Set the GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = torch.device("cuda:0")  # After setting CUDA_VISIBLE_DEVICES, cuda:0 maps to the correct GPU
    print(f"Client {client_id} running on {device}")

    # Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Client {client_id} connected to server")

    for message in messages:
        # Prepare data to send
        data = {
            "client_id": client_id,
            "text": message
        }

        # Serialize and send data to the server
        serialized_data = pickle.dumps(data)
        client_socket.sendall(len(serialized_data).to_bytes(4, byteorder='big'))  # Send data length
        client_socket.sendall(serialized_data)  # Send data
        print(f"Client {client_id} sent message: {message}")

        # Wait for acknowledgment from the server
        ack = client_socket.recv(1024).decode()
        if ack != "ACK":
            print(f"Client {client_id} received unexpected acknowledgment: {ack}")
        else:
            print(f"Client {client_id} received acknowledgment for message: {message}")

    # Close the socket
    client_socket.close()
    print(f"Client {client_id} finished processing")