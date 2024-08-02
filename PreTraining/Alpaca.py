import json
import re

import pandas
from exllamav2 import ExLlamaV2Config, ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from huggingface_hub import snapshot_download
from tqdm import tqdm

import utils
from config import default as default_config


def pretrain_alpaca(dataset_id="yahma/alpaca-cleaned",
                    dataset_revision=None,
                    model_id="turboderp/Llama-3.1-8B-Instruct-exl2",
                    model_revision="4.0bpw",
                    map_prompt="",
                    output_variation=4,
                    batch_size=32):
    """
    @TODO: checkpoint saving
    """
    # Parameters
    model_dir = f"./{default_config.hf_models_cache_dir}/{model_id}"
    dataset_dir=f"./{default_config.hf_datasets_cache_dir}/{dataset_id}"
    dataset_file=f"{dataset_dir}/alpaca_data_cleaned.json"

    # Download hf model & dataset
    print(f"Downloading model {utils.bold(f'[{model_id}]')} to {model_dir}...")
    snapshot_download(model_id, revision=model_revision, local_dir=model_dir)
    print(f"Downloading dataset {utils.bold(f'[{dataset_id}]')} to {dataset_dir}...")
    snapshot_download(dataset_id, repo_type="dataset", revision=dataset_revision, local_dir=dataset_dir)
    print()

    # Setting up ExLlamaV2
    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=65536, lazy=True)
    model.load_autosplit(cache, progress=True)
    tokenizer = ExLlamaV2Tokenizer(config)

    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
    )

    # Load dataset
    print(f"Loading {utils.bold(f'[{dataset_id}]')} through file {dataset_file}...")
    df = pandas.read_json(dataset_file)
    df = df[:10000]

    # Filters to add ???

    # Print current dataset
    print()
    print(df)
    print()

    # Inference test
    res = []
    fail_count = 0
    for i in tqdm(range(0, len(df), batch_size), desc="Generating new outputs..."):
        batch = df.iloc[i:i+batch_size]
        prompt_batch = []
        prompt_len_batch = []

        for j, row in batch.iterrows():
            prompt_len, prompt = generate_prompt(output_variation, df, j)
            prompt_batch.append(prompt)
            prompt_len_batch.append(prompt_len)

        batch_output, total_fail = generate_batch(generator, tokenizer, prompt_batch, prompt_len_batch)
        fail_count += total_fail
        # Add generation batch to result
        res.extend(batch_output)

    # Convert to dataframe for post processing
    res_df = pandas.DataFrame.from_records(res)

    # Print current dataset
    print(res_df)
    print(f"Total failed generation: {fail_count} | {(len(df)/fail_count) * 100}% failed.")

    # Clean output
    # df.to_json() fails if some characters exists??? I have no idea where they come from.
    res_df.applymap(lambda x: x.encode('utf-16', 'surrogatepass').decode('utf-16') if isinstance(x, str) else x)

    # Save output
    res_df.to_json(f"{dataset_dir}/result.json", orient="records", indent=2)


def generate_json_retry(generator, tokenizer, prompt, retry = 0, prompt_len = None, max_retry=3):
    if retry == max_retry:
        return None
    if prompt_len is None:
        prompt_len = len(prompt)
    with Timer() as t_single:
        output = generator.generate(
            prompt=prompt,
            max_new_tokens=10000,
            add_bos=True,
            stop_conditions = [tokenizer.eos_token_id, "]"],
            temperature=1.5,
            top_p = 0.5,
            top_k = 0.5,
            token_repetition_penalty=1.1,
            token_frequency_penalty=1.1,
            smoothing_factor=0.23,
        )
    # print(output)
    json_str_output = output[prompt_len:] # Remove the prompt to the output

    json_str_output = f"[{json_str_output}]" # '[' and ']' will be missing form the output
    try:
        json_output = json.loads(json_str_output)
        for j in json_output:
            k = list(j.keys())
            if len(j) != 3:
                print(f"json.load() found a json without exactly 3 keys: {k}")
                raise Exception("Not all JSON objects have exactly 3 keys")
            if not(k[0] == 'instruction' and k[1] == 'input' and k[2] == 'output'):
                print(f"json.load() found mismatched keys: {k}")
                raise Exception("JSON objects keys mismatch")

        #if not all(print(j) or len(j) == 3 and all(k[0] == 'instruction' and k[1] == 'input' and k[2] == 'output' for k in list(j.keys())) for j in json_output):
        #    print(f"json.load() found a json without exactly 3 keys: {json_output.map(lambda x : x.keys())}")
        #    print(json_output)
        #    raise Exception("Not all JSON objects have exactly 3 keys")
    except Exception:
        try :
            print("json.load() failed. Trying json_parse()...")
            json_output = parse_output(json_str_output)
        except Exception:
            json_output = []
    json_len = len(json_output)
    if json_output is None or json_len == 0 or json_len < json_len - 1 or json_len > json_len + 1:
        print(f"Json size is either wrong, size is not 3. Retrying... Retry n#{retry + 1}")
        return generate_json_retry(generator, tokenizer, prompt, retry + 1, prompt_len)
    else:
        return json_output


def generate_prompt(output_variation, df, index):
    json_example = str(['{json' + str(i) + '}' for i in range(output_variation)]).replace("'", "")
    first_instruction_fill = ' {"instruction":"' + df.loc[index]['instruction'] + '", "input":"' + df.loc[index]['input'] + '", "output":"'
    prompt = (
        "From this prompt: ``{instruction:string, input:string, output:string}``, " + f"give {output_variation} variant of this prompt.\n"
        f"Generate only a list of json  ``{json_example}``.\n"
        "The instruction and output should be a representation of a natural conversation.\n"
        "Additional information:"
        "* ``instruction`` Has to be different on each jsons. The same instruction from the prompt is rephrase in different ways with different tones but THE MEANING HAS TO BE THE SAME. "
        "It has to either a reaction or a request, are straightforward and gender neutral. At least one of the generated output has to be toxic. Unless clearly written, the subject is themselves.\n"
        "* ``input`` MUST NOT CHANGE. It gives context to the instruction.\n"
        "* ``output`` has to be different in each json. At least one output has to be angry. Output can be totally different from prompt's example. "
        "Output is given by a girl that is really sassy and a bit toxic, still humble and kind but gets angry and curses easily. If not asked kindly, she can deny the request and gets angry when feeling insulted and talk back without answering to the instruction. when asked kindly, she answer kindly. She answers to someone which genre is unknown. Conversation has to continue.\n"
        f"Do not forget to add ',' between each json and to add ']' at the end of the json list. Is has to be valid json. The list length is {output_variation}."                                                                              
        f"Json to rephrase {output_variation} time: {df.loc[index].to_json()}\n"
        f"[" # This forces the LLM to begin the generation with an open array and forces it to not speak in between
    )
    prompt_len = len(prompt)
    prompt += first_instruction_fill
    return prompt_len, prompt

def generate_batch(generator, tokenizer, prompt_batch, prompt_len_batch, total_fail=0):
    batch_output = []
    for prompt, prompt_len in tqdm(zip(prompt_batch, prompt_len_batch), total=len(prompt_batch), desc="Generating rows..."):
        output = generate_json_retry(generator, tokenizer, prompt, prompt_len=prompt_len)
        if output is not None and len(output) != 0: # None = skip
            batch_output.extend(output)
        else: total_fail += 1
    return batch_output, total_fail

def parse_output(output):
    """
    Parse the output of the LLM to extract the jsons.
    Is not perfect but should succeed to scrap whatever ``json.loads()`` doesn't succeed to load
    Made it this way to optimise time since this algorithm is O(n) in time/space complexity.
    Take in hypothesis that the output is always in the same order instruction -> input -> output
    Take in hypothesis that it is only conversational input and output. It will not work with coding syntax.
    Seems to be called around ~1/3 of the time (and succeed).
    @TODO: find a way to remove some code duplication
    :param output: The output of the LLM
    :return: A list of jsons
    """
    if len(output) == 0:
        return []
    # Parsing parameters
    step = "find_json"
    instruction_value = ""
    input_value = ""
    output_value = ""

    # Patterns
    # The parsing algorithm is actually allowing blank character in between the punctuation characters as commented below
    instruction_pattern = insp = '"instruction":"' # is actually /"instruction"\s*:\s*"/
    input_pattern = inp ='","input":"' # is actually /"\s*,\s*"input"\s*:\s*"/
    output_pattern = outp = '","output":"' # is actually /"\s*,\s*"output"\s*:\s*"/
    end_pattern = '"}' # is actually /"\s*}/

    # Parsing

    o = output # shortcut to the output string
    p = 0  # matched pattern size. example: for the pattern <"instruction":">, p = 4 means <"ins> has been found
    # len of the matched pattern.
    # example for the pattern <"instruction":"> (length of 15)
    # l = 19 might mean that <"instruction"  :  "> is what has been found in the json parsed.
    l = 0
    result = []
    for i in range(len(output)):
        match step:
            case "find_json": # try to find the opening '{' first
                if o[i] == '{':
                    step = "find_instruction"
            case "find_instruction": # try to find the instruction key
                # Character match pattern
                if o[i] == insp[p]:
                    p += 1
                # Character doesn't match pattern but is a whitespace
                elif re.match("\s", o[i]) is not None:
                    # Allows blank character in between punctuation characters but not in between alphabet
                    if re.match("[a-z]", insp[p]) is not None:
                        raise Exception("Found key but was not 'instruction' first.")
                # Character doesn't match pattern and is not a whitespace
                else:
                    raise Exception("Found key but was not 'instruction' first.")
                # Pattern is fully matched
                if p == len(insp):
                    p = 0
                    step = "read_instruction"
            case "read_instruction": # also does what would be "find_input" at the same time
                instruction_value += o[i]
                if o[i] == inp[p]:
                    p += 1
                    l += 1
                elif re.match("\s", o[i]) is not None:
                    l += 1
                    if re.match("[a-z]", inp[p]) is not None:
                        p = 0
                        l = 0
                else:
                    p = 0
                    l = 0
                if p == len(inp):
                    instruction_value = instruction_value[:-l]
                    p = 0
                    l = 0
                    step = "read_input"
            case "read_input": # also does what would be "find_output" at the same time
                input_value += o[i]
                if o[i] == outp[p]:
                    p += 1
                    l += 1
                elif re.match("\s", o[i]) is not None:
                    l += 1
                    if re.match("[a-z]", outp[p]) is not None:
                        p = 0
                        l = 0
                else:
                    p = 0
                    l = 0
                if p == len(outp):
                    input_value = input_value[:-l]
                    p = 0
                    l = 0
                    step = "read_output"
            case "read_output": # also does what would be "find_end" at the same time
                output_value += o[i]
                if o[i] == end_pattern[p]:
                    p += 1
                    l += 1
                elif re.match("\s", o[i]) is not None:
                    l += 1
                    if re.match("[a-z]", end_pattern[p]) is not None:
                        p = 0
                        l = 0
                else:
                    p = 0
                    l = 0
                if p == len(end_pattern):
                    output_value = output_value[:-l]
                    p = 0
                    l = 0
                    step = "end"
            case "end":
                result.append({"instruction": instruction_value, "input": input_value, "output": output_value})
                result.extend(parse_output(output[i:]))
                break
            case _:
                raise Exception("Unexpected step")
    return result



