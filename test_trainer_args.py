import transformers
print(transformers.__file__)
from transformers import TrainingArguments

print(TrainingArguments.__doc__)
print(TrainingArguments.__init__.__code__.co_varnames)

args = TrainingArguments(
    output_dir="./test",
    eval_strategy="steps"
)
print("Success! TrainingArguments accepted evaluation_strategy.") 