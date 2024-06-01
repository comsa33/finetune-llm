from datasets import load_dataset

def load_and_format_dataset(token, repo_name, eos_token, alpaca_prompt):
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction=instruction, input=input, response=output) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset(repo_name, split="train", token=token)
    return dataset.map(formatting_prompts_func, batched=True)
