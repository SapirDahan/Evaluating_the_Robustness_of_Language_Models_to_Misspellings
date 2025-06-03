import json
import csv
import os

# === FILE LOCATIONS ===
input_file = "ELI5-001.jsonl"  # in Data_Preparation/
output_file = os.path.join("..", "Data", "questions.csv")
num_questions = 50

# === ENSURE OUTPUT DIRECTORY EXISTS ===
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# === EXTRACT QUESTIONS ENDING WITH "?" ===
count = 0
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    writer = csv.writer(outfile)
    writer.writerow(['question'])  # CSV header

    for line in infile:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        question = item.get("question", "").strip()

        # Only take questions that end with '?'
        if question.endswith("?"):
            print(f"[{count+1}] {question}")
            writer.writerow([question])
            count += 1

        if count >= num_questions:
            break

print(f"\n✅ Done: Saved {count} question(s) ending with '?' to '../Data/questions.csv'")