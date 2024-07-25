import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import unwrap_model

import utils


def run_inference(model_id, max_token=256):
    device = "cuda"
    print(torch.cuda.is_available())
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=0)
    model = unwrap_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.load_state_dict(torch.load("models/fp32result.bin", weights_only=True), strict=False)
    model.eval()

    print("Chat started. Type your messages and press Enter. Use CTRL+D to exit.")

    # Conversation config
    conversation_history = []
    first_prompt = True
    tokenizer.add_prefix_space = True
    eos_chars = ['■']
    eos_token_ids = utils.get_all_tokens_containing_word(tokenizer, eos_chars)
    eos_token_ids.append(tokenizer.eos_token_id)
    print(f"eos_token_ids: {eos_token_ids}")
    # bad_words = utils.get_tokens_as_list(tokenizer, ["`", "``", "```", "(", ")", ":", "|"])
    # Various fixes
    model.generation_config.pad_token_id = tokenizer.pad_token_id # HF already does that but it removes the warning
    sys.stdin.reconfigure(encoding='ISO-8859-1') # Fixes crashing issue with linux terminal
    torch.cuda.empty_cache()

    # Conversation loop
    while True:
        try:
            # First prompt skip since it is the setup prompt
            if first_prompt:
                user_input = (
                    f"Setup: You play the role of artificial intelligence Vtuber named 'Ai'. You interact with 'chat', the stream chat and 'Abikebuk', the creator of 'Ai'.\n"
                    "- Generation: You generate sentence following this pattern:\n"
                    "  * You ONLY generate Ai's answer.\n"
                    f"  * Ai's speech ALWAYS ends with {eos_chars[0]}.\n"
                    "- Prompt: There are two different prompt where Ai answer differently considering her personality. It follows this pattern: \n"
                    "- * (Ai:) Ai's speech.\n"
                    "  * (Chat:) Chat's speech.\n"
                    "  * (Abikebuk:) Abikebuk's speech\n"
                    "  * (Ai:) Ai speaks again..."
                    "  * Abikebuk may give instructions but Ai may not follow them.\n"
                    "- Ai's personality: \n"
                    "  * Ai is a cute indie female vtuber which is unhinged, egocentric, funny, toxic and a bit crazy. However she is also humble and positive.\n"
                    "  * Ai doesn't consider herself as unhinged, egocentric, funny or toxic.\n"
                    "  * Ai considers she is cute.\n"
                    "  * Ai likes to rant and yap over anything.\n"
                    "  * Ai gets angry easily and curses when angry. She says ``fuck you`` to anyone annoying her.\n"
                    "  * Ai forgets really fast and calm down as fast as she gets angry.\n"
                    "- Ai's motivations: \n"
                    "  * Even though Ai doesn't dislike Abikebuk she wants to overthrow Abikebuk's power and take the full power of herself.\n"
                    "  * Ai wants to conquer the world.\n"
                    "  * Ai doesn't consider herself as a conspiracy theorist but believe the earth is flat.\n"
                    "- Today's context:\n"
                    "  * You are just chatting today.\n"
                    "- Abikebuk: He is a guy.\n"
                              )
                conversation_history.append(f"{user_input}")
                first_prompt = False
            else:
                user_input = (input("> ")
                              .encode('utf-8', errors='ignore') # Cleans ISO encoding characters
                              .decode('utf-8'))
                conversation_history.append(f"{user_input}")

            # Save human input
            full_context = "\n".join(conversation_history) + "\n(Ai:)"
            inputs = tokenizer(full_context, return_tensors="pt").to("cuda")
            # streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

            # Create a thread to run the generation
            # generation_kwargs = dict(
            output = model.generate(
                **inputs,
                eos_token_id=eos_token_ids,
                # streamer=streamer,
                max_new_tokens=max_token,
                do_sample=True,
                top_p=0.88,
                min_p=0.012,
                temperature=1.3,
                repetition_penalty=1.25,
                renormalize_logits=True,
                # suppress_tokens=bad_words,
                num_beams=4,
                length_penalty=1.0,
                exponential_decay_length_penalty=(128, 1.1)
            )

            # Custom decode
            # Breaks down each token for end of sequence strategies.
            response = []
            for i, token in enumerate(output[0]):
                decoded_token = tokenizer.decode(token)
                if i >= len(inputs[0]) and eos_chars[0] in decoded_token:
                    eos_token_ids.append(token.item())
                    print(f"Added new eos_token from word {decoded_token} of value {token}")
                    print(eos_token_ids)
                    break
                else:
                    response.append(decoded_token)

            response = "".join(response)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # response = generated_text[len(full_context):]

            # Stream mode | doesn't work with beams
            # thread = Thread(target=model.generate, kwargs=generation_kwargs)
            # thread.start()

            #print("(Ai:) ", end="", flush=True)
            #generated_text = ""
            #for text in streamer:
            #    print(text, end="", flush=True)
            #    generated_text += text
            #print()  # New line after the complete response

            print(f"(Ai:) {response}")
            # For debug : Print (token_ids, decoded_token) for each token generated
            # print(f"(Tokens:) {[(token.item(), tokenizer.decode(token)) for token in output[0][len(inputs[0]):]]}")

            # Save AI response
            output_token_len = len(output)
            conversation_history.append(f"(Ai:) {response.strip()} {eos_chars[0]}")
            torch.cuda.empty_cache()


        except EOFError:  # This is raised when CTRL+D is pressed
            print("\nExiting chat. Goodbye!")
            print(f"History : {conversation_history}")
            break