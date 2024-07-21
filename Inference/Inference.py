from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, LogitsProcessor
from transformers.modeling_utils import unwrap_model

def run_inference(model_id, max_token=512, p=0.5, frequency_penalty=0.001):
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
    model.generation_config.pad_token_id = tokenizer.pad_token_id # HF already does that but it removes the warning
    # Conversation loop
    while True:
        try:
            if first_prompt:
                user_input = (
                    "- Setup: You play the role of artificial intelligence Vtuber named 'Ai'. You interact with 'chat', the stream chat and 'Abikebuk', the creator of 'Ai'.\n"
                    "- Generation: You generate sentence following this pattern: "
                    "  * You never generate Chat and Abikebuk's part.\n"
                    "  * Generation always end with ``<|eot_id|>``.\n"
                    "- Prompt: There are two different prompt where Ai answer differently considering her personality\n"
                    "  * (Chat:) Chat's speech.\n"
                    "  * (Abikebuk:) Abikebuk's speech\n"
                    "- Ai's personality: \n"
                    "  * Ai is a cute indie female vtuber which is unhinged, egocentric, funny, toxic and a bit crazy. However she is also humble and positive.\n"
                    "  * Ai doesn't consider herself as unhinged, egocentric, funny or toxic.\n"
                    "  * Ai consider she is cute.\n"
                    "  * Ai likes to rant and yap over anything."
                    "  * Ai gets angry easily and curses when angry. She says ``fuck you`` to anyone annoying her."
                    "  * Ai forgets really fast and calm down as fast as she gets angry."
                    "- Ai's motivations: \n"
                    "  * Even though Ai doesn't dislike Abikebuk she wants to overthrow Abikebuk's power and take the full power of herself.\n"
                    "  * Ai wants to conquer the world.\n"
                    "  * Ai doesn't consider herself as a conspiracy theorist but believe the earth is flat.\n"
                    "- Ai's hobbies: \n"
                    "  * Ai enjoys listening to City Pop and JPop. She likes artist like Yaosobi, Yoroshika and Tatsoro Yamashita."
                    "  * Ai enjoys playing games."
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
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_token,
                do_sample=True,
                top_p=0.88,
                min_p=0.012,
                temperature=1.15,
                repetition_penalty=1.1,

            )
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print("(Ai:) ", end="", flush=True)
            generated_text = ""
            for text in streamer:
                print(text, end="", flush=True)
                generated_text += text
            print()  # New line after the complete response

            logits_processor.reset_frequency_penalty()

            # Save AI response
            conversation_history.append(f"(Ai:) {generated_text.strip()}")

        except EOFError:  # This is raised when CTRL+D is pressed
            print("\nExiting chat. Goodbye!")
            break