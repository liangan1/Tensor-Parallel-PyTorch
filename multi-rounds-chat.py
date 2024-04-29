from transformers.testing_utils import (
    is_torch_available,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
attn_implementation = "eager"

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,    
)
text = "how are you?" #len2text["32"]
inputs = tokenizer(text, padding=True, return_tensors="pt").to(model.device)

model.generation_config.cache_implementation = "dynamic"

with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True):
    gen_out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=10,
        early_stopping=True,
        eos_token_id= tokenizer.eos_token_id,
        num_beams=1,
        return_dict_in_generate=True,
    )
decoded = tokenizer.batch_decode(gen_out['sequences'], skip_special_tokens=True)
print("first round chat: output w/o kv_cache:", decoded)

past_key_values = gen_out["past_key_values"]

next_round_text = decoded[0] + " Thanks!, I am also busy with my work about LLM, can you help me with that?"
next_round_inputs =tokenizer(next_round_text, padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True):
    gen_out = model.generate(
        **next_round_inputs,
        do_sample=False,
        max_new_tokens=10,
        early_stopping=True,
        eos_token_id= tokenizer.eos_token_id,
        num_beams=1,
        return_dict_in_generate=True,
        past_key_values=past_key_values,
    )
decoded_w_kv_cache = tokenizer.batch_decode(gen_out['sequences'], skip_special_tokens=True)
print("second round chat: output w/ kv_cache: ", decoded_w_kv_cache)
with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True):
    gen_out = model.generate(
        **next_round_inputs,
        do_sample=False,
        max_new_tokens=10,
        early_stopping=True,
        eos_token_id= tokenizer.eos_token_id,
        num_beams=1,
        return_dict_in_generate=True,
        #past_key_values=past_key_values,
    )
decoded_wo_kv_cache = tokenizer.batch_decode(gen_out['sequences'], skip_special_tokens=True)
print("second round chat: output w/o kv_cache: ", decoded_wo_kv_cache)
