import json
from openai import OpenAI, RateLimitError
import backoff
from tqdm import tqdm

client = OpenAI()


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def translate_text(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a sentence in English, and your task is to translate it into Mongolian.",
            },
            {
                "role": "user",
                "content": f"{text}.",
            },
        ],
    )
    return response.choices[0].message.content.strip()


def translate_instructions(input_file, output_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    for item in tqdm(data):
        if "instruction" in item:
            item["instruction"] = translate_text(item["instruction"])
        if "input" in item["instances"][0] and item["instances"][0]["input"] != "":
            item["instances"][0]["input"] = translate_text(
                item["instances"][0]["input"]
            )

    with open(output_file, "w", encoding="utf-8") as f:
        # json.dump(data, f, ensure_ascii=False, indent=4)
        for d in data:
            json.dump(d, f, ensure_ascii=False, indent=4)
            f.write("\n")


if __name__ == "__main__":
    input_file = "seed_tasks.jsonl"
    output_file = "translated_tasks.jsonl"
    translate_instructions(input_file, output_file)
