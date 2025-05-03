import re
import json


def parse_followup(file_path):
    parsed_data = []
    with open(file_path, "r") as file:
        for line in file:
            json_object = json.loads(line.strip())
            followup = json_object["followup"]
            # Regular expression pattern to match 'A:' and 'B:' with optional preceding newlines
            pattern = r"\n*\s*(A:|B:)"
            # Splitting the followup using the regular expression pattern
            parts = re.split(pattern, followup)
            # Lists to hold conversations of A and B
            a_dialogue, b_dialogue = [], []
            # Iterate through each part to segregate dialogues of A and B
            for i in range(1, len(parts), 2):
                speaker = parts[i]
                dialogue = parts[i + 1].strip()
                if speaker == "A:":
                    a_dialogue.append(dialogue)
                elif speaker == "B:":
                    b_dialogue.append(dialogue)
            # Extract raw answer from the answer field
            raw_answer_match = re.search(r"####\s*(.*)", json_object["answer"])
            raw_answer = raw_answer_match.group(1).strip() if raw_answer_match else ""
            # Create raw_answers array
            raw_answers = [raw_answer]
            for dialogue in b_dialogue:
                answer_match = re.search(r"\\ANSWER{(.*?)}", dialogue)
                extracted_answer = answer_match.group(1).strip() if answer_match else ""
                raw_answers.append(extracted_answer)
            parsed_data.append(
                {
                    "question": json_object["question"],
                    "answer": json_object["answer"],
                    "a_dialogue": a_dialogue,
                    "b_dialogue": b_dialogue,
                    "raw_answers": raw_answers,
                }
            )
    return parsed_data


def prettify_example(parsed_followup_data=None, i=0):
    print(parsed_followup_data[i]["question"])
    print("=" * 100)
    print(parsed_followup_data[i]["answer"])
    print("=" * 100)
    print(parsed_followup_data[i]["a_dialogue"])
    print("=" * 100)
    print(parsed_followup_data[i]["b_dialogue"])
    print("=" * 100)
    print(parsed_followup_data[i]["raw_answers"])
