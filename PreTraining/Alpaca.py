import json

import pandas
from exllamav2 import ExLlamaV2Config, ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from huggingface_hub import snapshot_download

import utils
from config import default as default_config


def pretrain_alpaca(dataset_id="yahma/alpaca-cleaned", dataset_revision=None, model_id="turboderp/Llama-3.1-8B-Instruct-exl2", model_revision="4.0bpw", map_prompt="", output_variation=4):
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
    print()
    print(df)
    print()

    # Inference test
    prompt = map_prompt
    prompt = (
        "From this prompt: ``{instruction:string, input:string, output:string}``, " + f"give {output_variation} variant of this prompt.\n"
        f"Generate only a list of json  ``{['{json' + str(i) + '}' for i in range(output_variation)]}``. Prettify the generation.\n"
        "The instruction and output should be a representation of a natural conversation."
        "Additional information:"
        "* ``instruction`` The same instruction from the prompt is reformulated in different ways with different emotions but THE MEANING HAS TO BE THE SAME. "
        "It has to either a reaction or a request, are straightforward and gender neutral. At least one of the generated output has to be toxic. Unless clearly written, the subject is themselves.\n"
        "* ``input`` MUST NOT CHANGE. It gives context to the instruction.\n"
        "* ``output`` has to be different in each json. At least one output has to be angry. Output is aimed toward a person which genre is unknown. Output can be totally different from prompt's example. "
        "Output is given by a girl that is really sassy and a bit toxic, still humble and kind but gets angry and curses easily. If not asked kindly, she can deny the request and gets angry when insulted.\n"
        f"{df.loc[0].to_json()}\n"
        f"[\n"
    )


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
    print(f"{output}]")
