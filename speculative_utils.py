# speculative_utils.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import socket
import pickle
import numpy as np

# --- Model Setup ---
def load_model(model_name, device):
    """Load the specified model and move it to the given device."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model

def load_tokenizer():
    """Load shared tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# --- Count Words in a Sentence ---
def count_words(text):
    """Count the number of words in a text string."""
    return len(text.split())

# --- Count Tokens in a Sentence ---
def count_tokens(text, tokenizer):
    """Count the number of tokens in a text string using the tokenizer."""
    tokens = tokenizer.encode(text, return_tensors="pt")
    return tokens.shape[1]

# --- Probability Distribution Function ---
def get_distribution(model, input_ids, temperature=1.0):
    """Get probability distribution over next token from model."""
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        return probs, token_id

# --- Simplified Speculative Sampling Function ---
def speculative_sample(primal_probs, draft_probs, draft_token):
    """Simplified speculative sampling to accept/reject draft guess."""
    if isinstance(draft_token, torch.Tensor):
        draft_token = draft_token.item()

    p_x = primal_probs[0, draft_token].item()
    q_x = draft_probs[0, draft_token].item()
    r = np.random.uniform(0, 1)
    if r <= p_x / q_x:
        return draft_token, None
    else:
        adjusted_probs = torch.max(primal_probs[0] - draft_probs[0], torch.zeros_like(draft_probs[0]))
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        new_token = torch.multinomial(adjusted_probs, num_samples=1).item()
        return None, new_token

# --- Draft Model Process (Client) ---
def draft_model_process(model_name, gpu_index, draft_id, prompts, host='localhost', port=5000, temperature=1.0):
    """Draft model process that generates speculative tokens and sends them to the primal model server."""
    # Set the device directly using the global GPU index
    device = torch.device(f"cuda:{gpu_index}")
    print(f"Draft {draft_id} running on cuda:{gpu_index}")  # Log the global GPU index

    # Load model and tokenizer
    model = load_model(model_name, device)
    tokenizer = load_tokenizer()

    # Connect to the primal model server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Draft {draft_id} connected to primal model server")

    # Define maximum token length for the final output
    max_token_length = 40  # Hardcoded maximum token length

    for prompt in prompts:
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_token_length = input_ids.shape[1]

        # Generate speculative tokens until reaching half of max_token_length or 30 words
        draft_guesses = []
        current_ids = input_ids.clone()
        current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        word_count = count_words(current_text)

        # Target half of max_token_length to leave room for primal model
        target_tokens = max_token_length // 2
        while current_ids.shape[1] - prompt_token_length < target_tokens and word_count < 30:
            draft_probs, draft_token = get_distribution(model, current_ids, temperature)
            draft_guesses.append((draft_token, draft_probs.tolist()))
            current_ids = torch.cat([current_ids, torch.tensor([[draft_token]], device=device)], dim=1)
            current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
            word_count = count_words(current_text)

        # Prepare data to send
        data = {
            "draft_id": draft_id,
            "prompt": prompt,
            "input_ids": input_ids.tolist(),
            "guesses": draft_guesses
        }

        # Serialize and send data to the primal model
        serialized_data = pickle.dumps(data)
        client_socket.sendall(len(serialized_data).to_bytes(4, byteorder='big'))  # Send data length
        client_socket.sendall(serialized_data)  # Send data
        print(f"Draft {draft_id} sent prompt: {prompt}")

        # Wait for the final output from the primal model
        data_length_bytes = client_socket.recv(4)
        data_length = int.from_bytes(data_length_bytes, byteorder='big')
        data = b""
        while len(data) < data_length:
            packet = client_socket.recv(data_length - len(data))
            data += packet
        final_output = pickle.loads(data)
        print(f"Draft {draft_id} received final output for prompt '{prompt}': {final_output}")

    # Close the socket
    client_socket.close()
    print(f"Draft {draft_id} finished processing")

# --- Primal Model Process (Server) ---
def primal_model_process(model_name, gpu_index, output_queue, host='localhost', port=5000, temperature=1.0):
    """Primal model process that serves as a server to verify or reject speculative tokens."""
    # Set the device directly using the global GPU index
    device = torch.device(f"cuda:{gpu_index}")
    print(f"Primal model running on cuda:{gpu_index}")  # Log the global GPU index

    # Load model and tokenizer
    model = load_model(model_name, device)
    tokenizer = load_tokenizer()

    # Set up the server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(2)  # Listen for 2 draft models
    print(f"Primal model server started, listening on {host}:{port}")

    # Accept connections from draft models
    clients = []
    for _ in range(2):  # Expecting 2 draft models
        client_socket, addr = server_socket.accept()
        clients.append((client_socket, addr))
        print(f"Primal model accepted connection from {addr}")

    # Define maximum token length for the final output
    max_token_length = 40  # Hardcoded maximum token length

    # Process requests in FIFO order
    final_outputs = []
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
                request = pickle.loads(data)
                draft_id = request["draft_id"]
                prompt = request["prompt"]
                input_ids = torch.tensor(request["input_ids"], device=device)
                draft_guesses = request["guesses"]

                # Process the speculative guesses
                accepted_tokens = []
                current_ids = input_ids.clone()
                for draft_token, draft_probs in draft_guesses:
                    draft_probs = torch.tensor(draft_probs, device=device)
                    primal_probs, _ = get_distribution(model, current_ids, temperature)
                    accepted_token, rejected_token = speculative_sample(primal_probs, draft_probs, draft_token)
                    if accepted_token is not None:
                        accepted_tokens.append(accepted_token)
                        current_ids = torch.cat([current_ids, torch.tensor([[accepted_token]], device=device)], dim=1)
                    else:
                        accepted_tokens.append(rejected_token)
                        current_ids = torch.cat([current_ids, torch.tensor([[rejected_token]], device=device)], dim=1)
                        break

                # Continue generating tokens until the output reaches 30-40 words or max_token_length
                current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                word_count = count_words(current_text)
                prompt_token_length = input_ids.shape[1]
                while word_count < 30 and current_ids.shape[1] < max_token_length:
                    primal_probs, extra_token = get_distribution(model, current_ids, temperature)
                    current_ids = torch.cat([current_ids, torch.tensor([[extra_token]], device=device)], dim=1)
                    current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                    word_count = count_words(current_text)

                # Decode the final output
                final_output = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                final_outputs.append((draft_id, prompt, final_output))
                print(f"Primal model processed prompt from Draft {draft_id}: {prompt}")

                # Send the final output back to the draft model
                serialized_output = pickle.dumps(final_output)
                client_socket.sendall(len(serialized_output).to_bytes(4, byteorder='big'))  # Send data length
                client_socket.sendall(serialized_output)  # Send data

            except (ConnectionResetError, BrokenPipeError):
                active_clients -= 1
                print(f"Connection closed by Draft {draft_id}")
                continue

    # Close all client sockets and the server socket
    for client_socket, _ in clients:
        client_socket.close()
    server_socket.close()
    print("Primal model server shut down")

    # Send the final outputs back to the main process via the queue
    output_queue.put(final_outputs)
