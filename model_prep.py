from unsloth import FastLanguageModel

def prepare_model(model_name, max_seq_length, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer
