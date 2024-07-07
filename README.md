# Intelligence
This project is just a repo where I keep a trace of everything I am learning.  

## What if you stumble upon this repo
I hope you don't expect anything but if you are have any question or are willing to give me some tips you can reach me on Discord: ``Abikebuk#2009 ``

## Goal
My main goal is to train a LLM, Llama3, with reinforcement learning and be able to feed it data from the web through web APIs or maybe giving it access to a web browser.

## Movitvations
I want to learn AI. I am basically beginning rom scratch. I don't know if this repo will results in anything meaningful but i'll try my best.

## What I need?
List of things I need to do to reach my goal.  
*This list is based on "what I understand" until now and surely unoptimal or maybe inacurrate on what I should do.*

* Get a labeled dataset
* Since I don't plan to self source some data, I won't have labels to train data.
* Use BERT to classify data and make the labels for me
* Being able to train BERT
* Train Llama3 with my dataset
* Train Llama3 with reinforcement learning
* With the help of BERT, transform my self-sourced data to use with Llama3 through reinforcement learning

## Limitations
* I use a good laptop but only own a RTX 3070 which contains only 8gb of VRAM and might not suitable for AI. In that case, I am trying to use lower quntized model and other workaround to make it work in my case.
* I don't want to pay any gpu cloud services (they are expensive). I am considering buying a GPU but I am reluctant to that idea.

## Dependency hell
Dependencies are fulfilled by running:
```bash
# Create the conda environment
conda create -n intelligence python=3.10
conda activate installigence

# Install dependencies
pip install -r requirement.txt

# But somehow it won't install torch first so if failed, try to run in this instead:
pip install torch
pip install -r requirement.txt
```
Deepspeed made the task a bit more complicated. Those are for Arch:
```bash
# Get cuda 
sudo pacman -S cuda
# Add missing file, heard it is a conda issue. Change intelligence with whatever env name you use
ln -s /opt/cuda/lib64/libcurand.so /opt/anaconda/envs/intelligence/lib/python3.10/site-packages/torch/lib/
# Install other missing library for deepspeed
conda install -c conda-forge gcc
# Compile Deepspeed's Adam-cpu
python -c 'import deepspeed; deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()'
```
Took me a while to find how to make it work but it is functional now.

## Objective-list
* Learning Pytorch
  * [x] Find a way to use HuggingFace's model/dataset with Pytorch
  * [ ] Learn to use ``torch.nn``
* BERT
  * [x] Find the right BERT type model to use.  
    - Chose ``distilbert/distilroberta-base``   
  * [x] Train Roberta with MLM (Maskeked Language Modeling)
  * [ ] Perfom inference with bert
  * [ ] Verify the training goes in the direction I want
  * [x] Classify data with bert
    * Classification works fine but to be tested
* Llama3
  * [x] Find the right Llama3 model to use.
    - Chose ``astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit``
    - Changed it to ``ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16`` instead. I don't have enough GPU VRAM to train the previous model.
  * [x] Train llama3
    - Would succeed to train on with a batch size of 1. It would go twice slower than a batch size of 2 but I don't have enough VRAM.
  * [ ] Optimize training
    *  [x] Find a way to reduce VRAM usage
      - Used Deepspeed to decrease VRAM usage by a decent amount. I was able to train only a batch size of 4. Now it has increased to 6.
    *  [x] Find a way to reduce training time
      - Deepspeed on stage 2 is literally twice faster.
    *  [ ] Find a way to increase training quality
  * [ ] Verify the training goes in the direction I want
  * [ ] Train llama3 with reinforcement learning


## References
* [HunggingFace](https://huggingface.co/)
  * [distilbert/distilroberta-base](https://huggingface.co/distilbert/distilroberta-base)
  * [astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit](https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit)
  * [ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16)
  * [alpindale/light-novels](https://huggingface.co/datasets/alpindale/light-novels)
  * [Sentiment Analysis](https://huggingface.co/blog/sentiment-analysis-twitter)
* [PyTorch](https://pytorch.org/docs/stable/index.html)
* [Text classification example by Claude Feldges](https://medium.com/@claude.feldges/text-classification-with-bert-in-tensorflow-and-pytorch-4e43e79673b3)
* [Deepspeed](https://github.com/microsoft/DeepSpeed)
* Used some AIs
  * [Anthropics Claude Haiku](https://www.anthropic.com/)
  * [Refact.ai](https://www.refact.ai/)

## Credit
* @Quinntyx for giving me the repo name idea which is a direct reference to Apple Intelligence.