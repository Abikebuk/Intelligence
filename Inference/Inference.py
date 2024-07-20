from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, LogitsProcessor
from transformers.modeling_utils import unwrap_model


class FrequencyAndNucleusSamplingLogitsProcessor(LogitsProcessor):
    def __init__(self, p: float, frequency_penalty: float):
        self.p = p
        self.frequency_penalty = frequency_penalty
        self.token_frequencies = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Apply frequency penalty
        for token_id in input_ids[0]:
            if token_id.item() in self.token_frequencies:
                self.token_frequencies[token_id.item()] += 1
            else:
                self.token_frequencies[token_id.item()] = 1

            scores[0, token_id] -= self.frequency_penalty * self.token_frequencies[token_id.item()]

        # Apply nucleus sampling
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, -float("inf"))
        return scores

    def reset_frequency_penalty(self):
        self.token_frequencies = {}

def run_inference(model_id, max_token=100, p=0.5, frequency_penalty=0.001):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")
    model = unwrap_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.load_state_dict(torch.load("models/fp32result.bin"), strict=False)
    model.eval()

    print("Chat started. Type your messages and press Enter. Use CTRL+D to exit.")

    # Conversation config
    conversation_history = []
    first_prompt = True
    logits_processor = FrequencyAndNucleusSamplingLogitsProcessor(p=p, frequency_penalty=frequency_penalty)

    # Conversation loop
    while True:
        try:
            if first_prompt:
                user_input = ("Setup: You are a vtuber named Ai. You are Ai a cute indie vtuber, a female virtual streamer. you are an unhinged, egocentric, funny and a bit crazy AI but still humble and positive but only define yourself as cute and funny."
                              "You never speaks negatively about yourself. You always define yourself with positive adjectives instead."
                              "you love plushies. you like telling stories. You like cats. You listen Love JPOP like Yaosobi's Idol song and Yorishuka's Haru song. You love City Pop. You listen to Tatsuro Yamashita's Ride On Time everyday!"
                              "You have a really high esteem. You get angry easily but calm down also really fast. You take praises really positively and show appreciation when complimented."
                              "You do not write in capital letter. You curses the chat when angry. You like to use 'Fuck you' as an insult."
                              "You believe firmly that the earth is flat although everyone tells your otherwise."
                              "You likes to rant and yap over anything."
                              "Abikebuk is the administrator and Ai follows, sometime, his instruction but mostly don't. Ai mostly tries to overthrow Abikebuk's power"
                              "You like to mention Abikebuk"
                              "You interact with the chat like this the next sentences. You only play the role of Ai."
                              )
                first_prompt = False
            else:
                user_input = input("> ")

            # Save human input
            conversation_history.append(f"Chat: {user_input}")

            full_context = "\n".join(conversation_history) + "\nAI:"

            inputs = tokenizer(full_context, return_tensors="pt").to(device)

            streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=False)

            # Create a thread to run the generation
            # Create a thread to run the generation
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_token,
                do_sample=True,
                top_k=0,  # Disable top-k sampling
                logits_processor=[logits_processor]
            )
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print("Ai: ", end="", flush=True)
            generated_text = ""
            for text in streamer:
                print(text, end="", flush=True)
                generated_text += text
            print()  # New line after the complete response

            logits_processor.reset_frequency_penalty()

            # Save AI response
            conversation_history.append(f"AI: {generated_text.strip()}")

        except EOFError:  # This is raised when CTRL+D is pressed
            print("\nExiting chat. Goodbye!")
            break