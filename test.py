import torch
from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

from prompts import alpaca_prompt

# 체크포인트 디렉토리 설정
checkpoint_dir = 'outputs/checkpoint-1000'

# 체크포인트에서 모델과 토크나이저 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_dir,
    max_seq_length=4096,
    dtype=None,  # 체크포인트 저장 시 사용된 데이터 타입 설정
    load_in_4bit=True  # 체크포인트 저장 시 사용된 설정
)


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return torch.any(input_ids == self.stop_token_id)


stop_token = "<|end_of_text|>"
stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)[0]

stopping_criteria = StoppingCriteriaList(
    [StopOnToken(stop_token_id)]
)


question_level = 3
question_type = {
    "name_eng": "Vocabulary-focused questions",
    "description": "이 유형의 문제는 단어의 의미, 사용 방법, 문맥 이해 등에 대한 이해를 물어봅니다. 각 세부 유형은 특정 단어 유형(예: 명사, 동사, 형용사 등)에 초점을 맞춥니다. 오답 선택지에는 같은 품사이면서 의미가 유사하지만 문맥에 맞지 않는 단어가 포함될 수 있습니다."
}
question_subtype = {
    "name_eng": "Adjectives",
    "description": "이 유형의 문제는 형용사의 적절한 선택과 사용을 물어봅니다. 형용사는 명사나 대명사를 수식하여 그것의 성질, 상태, 양 등을 나타냅니다."
}

FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction=f"""Generate a quiz of TOEIC Part 5 complying with the following quiz level(1 ~ 5) and question type, subtype.
                question_level: {question_level}, 
                question_type: {question_type['name_eng']} ({question_type['description']}),
                question_subtype: {question_subtype['name_eng']} ({question_subtype['description']})
                "You must follow the json format given.""",
            input="""You must follow the json format given.
                question_text: The question text of the quiz.
                choices: The list of 4 choices."
                correct_answer: The correct answer of the quiz."
                translation: The translation of the question text in Korean."
                explanation: The explanation of the quiz in Korean."
                vocabularies: The list of json objects of vocabularies in the question text and choices. The attributes are as follows."
                   word: The word of the vocabulary."
                   translation: The translation of the vocabulary."
                   difficulty: The difficulty of the vocabulary."
                   explanation: The explanation of the vocabulary."
                   part_of_speech: The part of speech of the vocabulary."
                   example: The example of the vocabulary."
                   example_translation: The translation of the example in Korean.""",
            response=""
        )
    ],
    return_tensors="pt",
).to("cuda")

if __name__ == "__main__":

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=4096,
        stopping_criteria=stopping_criteria
    )