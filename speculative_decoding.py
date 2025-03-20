# # speculative_decoding.py
# import multiprocessing as mp
# import time
# import torch
# from speculative_utils import primal_model_process, draft_model_process

# # --- Main Function ---
# def main(num_speculative_tokens=3):
#     # Verify GPU availability
#     if torch.cuda.device_count() < 3:
#         raise RuntimeError("At least 3 GPUs are required. Found only {}.".format(torch.cuda.device_count()))

#     # Define GPU indices
#     gpu_index_primal = '0'
#     gpu_index_draft1 = '1'
#     gpu_index_draft2 = '2'

#     # Define prompts for draft models
#     draft1_prompts = [
#         "The cat sleeps peacefully in the sun",
#         "A gentle rain falls on the quiet village",
#         "She dances with grace under the moonlight",
#         "The old oak tree stands tall in the forest",
#         "Children laugh and play in the park",
#         "A stormy sea crashes against the cliffs",
#         "He writes a letter by candlelight",
#         "The eagle soars high above the mountains",
#         "Flowers bloom in the spring garden",
#         "A train whistle echoes through the valley",
#         "The chef prepares a delicious meal",
#         "Stars twinkle in the clear night sky",
#         "A river flows quietly through the plains",
#         "She sings a song of love and loss",
#         "The lion roars in the savannah",
#         "Snowflakes fall softly on the rooftops",
#         "He paints a picture of the countryside",
#         "The market buzzes with morning activity",
#         "A fox darts across the frozen lake",
#         "The poet dreams of distant lands"
#     ]
#     draft2_prompts = [
#         "Dogs howl at the full moon tonight",
#         "A warm breeze rustles the palm trees",
#         "He runs through the crowded city streets",
#         "The desert stretches endlessly under the sun",
#         "Laughter fills the room during the party",
#         "Thunder rumbles over the dark horizon",
#         "She knits a sweater by the fireplace",
#         "The hawk circles above the open field",
#         "Roses glow red in the summer air",
#         "A ship sails toward the distant shore",
#         "The baker kneads dough at dawn",
#         "Comets streak across the midnight sky",
#         "A brook trickles down the hillside",
#         "He plays a melody on the old piano",
#         "The tiger stalks prey in the jungle",
#         "Icicles gleam on the winter cabin",
#         "She sculpts a figure from wet clay",
#         "The festival lights up the town square",
#         "A deer leaps over the woodland stream",
#         "The explorer maps uncharted territory"
#     ]

#     # Create a queue to retrieve final outputs from the primal model process
#     output_queue = mp.Queue()

#     # Start timing
#     start_time = time.perf_counter()

#     # Create processes for primal and draft models
#     primal_proc = mp.Process(target=primal_model_process, args=("gpt2-large", gpu_index_primal, output_queue, 'localhost', 5000, num_speculative_tokens))
#     draft1_proc = mp.Process(target=draft_model_process, args=("gpt2", gpu_index_draft1, 1, draft1_prompts, 'localhost', 5000, num_speculative_tokens))
#     draft2_proc = mp.Process(target=draft_model_process, args=("gpt2", gpu_index_draft2, 2, draft2_prompts, 'localhost', 5000, num_speculative_tokens))

#     # Start all processes
#     primal_proc.start()
#     # Give the primal model server a moment to start
#     time.sleep(1)
#     draft1_proc.start()
#     draft2_proc.start()

#     # Wait for draft models to finish
#     draft1_proc.join()
#     draft2_proc.join()

#     # Wait for primal model to finish
#     primal_proc.join()

#     # Retrieve final outputs from the queue
#     final_outputs = output_queue.get()

#     # End timing
#     end_time = time.perf_counter()
#     total_time = end_time - start_time

#     # Print final outputs
#     print("\nFinal Outputs After Speculative Decoding:")
#     for draft_id, prompt, output in final_outputs:
#         print(f"Draft {draft_id} Prompt: {prompt}")
#         print(f"Output: {output}\n")

#     print(f"Total execution time: {total_time:.2f} seconds")

# if __name__ == "__main__":
#     mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
#     main(num_speculative_tokens=7)

# speculative_decoding.py
import multiprocessing as mp
import time
import torch
from speculative_utils import primal_model_process, draft_model_process, load_tokenizer, count_words, count_tokens

# --- Main Function ---
def main():
    # Verify GPU availability
    if torch.cuda.device_count() < 3:
        raise RuntimeError("At least 3 GPUs are required. Found only {}.".format(torch.cuda.device_count()))

    # Define GPU indices
    gpu_index_primal = 0
    gpu_index_draft1 = 1
    gpu_index_draft2 = 2

    # Load tokenizer for verification
    tokenizer = load_tokenizer()

    # Define prompts for draft models
    draft1_prompts = [
        "The cat sleeps peacefully in the sun",
        "A gentle rain falls on the quiet village",
        "She dances with grace under the moonlight",
        "The old oak tree stands tall in the forest",
        "Children laugh and play in the park",
        "A stormy sea crashes against the cliffs",
        "He writes a letter by candlelight",
        "The eagle soars high above the mountains",
        "Flowers bloom in the spring garden",
        "A train whistle echoes through the valley",
        "The chef prepares a delicious meal",
        "Stars twinkle in the clear night sky",
        "A river flows quietly through the plains",
        "She sings a song of love and loss",
        "The lion roars in the savannah",
        "Snowflakes fall softly on the rooftops",
        "He paints a picture of the countryside",
        "The market buzzes with morning activity",
        "A fox darts across the frozen lake",
        "The poet dreams of distant lands"
    ]
    draft2_prompts = [
        "Dogs howl at the full moon tonight",
        "A warm breeze rustles the palm trees",
        "He runs through the crowded city streets",
        "The desert stretches endlessly under the sun",
        "Laughter fills the room during the party",
        "Thunder rumbles over the dark horizon",
        "She knits a sweater by the fireplace",
        "The hawk circles above the open field",
        "Roses glow red in the summer air",
        "A ship sails toward the distant shore",
        "The baker kneads dough at dawn",
        "Comets streak across the midnight sky",
        "A brook trickles down the hillside",
        "He plays a melody on the old piano",
        "The tiger stalks prey in the jungle",
        "Icicles gleam on the winter cabin",
        "She sculpts a figure from wet clay",
        "The festival lights up the town square",
        "A deer leaps over the woodland stream",
        "The explorer maps uncharted territory"
    ]

    # Create a queue to retrieve final outputs from the primal model process
    output_queue = mp.Queue()

    # Start timing
    start_time = time.perf_counter()

    # Create processes for primal and draft models
    primal_proc = mp.Process(target=primal_model_process, args=("gpt2-large", gpu_index_primal, output_queue))
    draft1_proc = mp.Process(target=draft_model_process, args=("gpt2", gpu_index_draft1, 1, draft1_prompts))
    draft2_proc = mp.Process(target=draft_model_process, args=("gpt2", gpu_index_draft2, 2, draft2_prompts))

    # Start all processes
    primal_proc.start()
    # Give the primal model server a moment to start
    time.sleep(1)
    draft1_proc.start()
    draft2_proc.start()

    # Wait for draft models to finish
    draft1_proc.join()
    draft2_proc.join()

    # Wait for primal model to finish
    primal_proc.join()

    # Retrieve final outputs from the queue
    final_outputs = output_queue.get()

    # End timing
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Print final outputs with word count verification
    print("\nFinal Outputs After Speculative Decoding:")
    for draft_id, prompt, output in final_outputs:
        word_count = count_words(output)
        print(f"Draft {draft_id} Prompt: {prompt}")
        print(f"Output: {output}")
        print(f"Word Count: {word_count}\n")

    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
    main()