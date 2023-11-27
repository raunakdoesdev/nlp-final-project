from datasets import load_dataset
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

class AnthropicModel:
    def __init__(self, model):
        self.anthropic = Anthropic()
        self.model = model

    def generate_guess(self, conversation,  question):
        completion = self.anthropic.completions.create(
            model=self.model,
            max_tokens_to_sample=300,
            prompt=f"{HUMAN_PROMPT}{question}{AI_PROMPT}A:",
        )

    def __call__(self, prompt):
        return self.model(prompt)



dataset = load_dataset("MemGPT/MSC-Self-Instruct")

previous_dialogues = dataset['train']['previous_dialogs'][0]

client = OpenAI(
  organization='YOUR_ORG_ID',
)

for row, dialogue in enumerate(dataset['train']['previous_dialogs']):
    full_dialogue = []
    for d in dialogue:
        conversation = d['dialog']
        for i, turn in enumerate(conversation):
            speaker = 'A' if i % 2 == 0 else 'B'
            full_dialogue.append(f"{speaker}: \"{turn['text']}\"\n")


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Based on the following dialogue:"
        },
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)


with open('dialogues.txt', 'w') as f: