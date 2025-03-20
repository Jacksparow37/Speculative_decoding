# socket_interaction.py
import multiprocessing as mp
import time
import torch
from socket_utils import server_process, client_process

# --- Main Function ---
def main():
    # Verify GPU availability
    if torch.cuda.device_count() < 3:
        raise RuntimeError("At least 3 GPUs are required. Found only {}.".format(torch.cuda.device_count()))

    # Define GPU indices
    gpu_index_server = 0
    gpu_index_client1 = 1
    gpu_index_client2 = 2

    # Define messages for clients
    client1_messages = [f"Client 1 Message {i}" for i in range(5)]
    client2_messages = [f"Client 2 Message {i}" for i in range(5)]

    # Create a queue to retrieve received messages from the server process
    output_queue = mp.Queue()

    # Start timing
    start_time = time.perf_counter()

    # Create processes for server and clients
    server_proc = mp.Process(target=server_process, args=(gpu_index_server, output_queue))
    client1_proc = mp.Process(target=client_process, args=(gpu_index_client1, 1, client1_messages))
    client2_proc = mp.Process(target=client_process, args=(gpu_index_client2, 2, client2_messages))

    # Start all processes
    server_proc.start()
    # Give the server a moment to start
    time.sleep(1)
    client1_proc.start()
    client2_proc.start()

    # Wait for clients to finish
    client1_proc.join()
    client2_proc.join()

    # Wait for server to finish
    server_proc.join()

    # Retrieve received messages from the queue
    received_messages = output_queue.get()

    # End timing
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Print received messages
    print("\nMessages Received by Server:")
    for client_id, message in received_messages:
        print(f"From Client {client_id}: {message}")

    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
    main()