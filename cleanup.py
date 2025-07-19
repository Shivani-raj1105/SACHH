import os
import shutil
import csv
import re

def ask_and_delete(path, is_dir=False):
    if os.path.exists(path):
        resp = input(f"Delete {path}? (y/n): ").strip().lower()
        if resp == 'y':
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"Deleted {path}")
        else:
            print(f"Kept {path}")

def clean_feedback_csv():
    input_file = 'feedback.csv'
    output_file = 'feedback_cleaned.csv'
    rows = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            conf = row['confidence']
            # If confidence is a percentage string, convert to float
            if isinstance(conf, str) and conf.strip().endswith('%'):
                try:
                    conf_float = float(conf.strip().replace('%','')) / 100.0
                    row['confidence'] = str(conf_float)
                except Exception:
                    row['confidence'] = ''
            rows.append(row)
    # Write cleaned data back (overwrite original file)
    with open(input_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['text','model_prediction','confidence','explanation','feedback_type'])
        writer.writeheader()
        writer.writerows(rows)
    print('feedback.csv cleaned!')

if __name__ == '__main__':

    checkpoints_dir = './bert-fake-news'
    if os.path.isdir(checkpoints_dir):
        for name in os.listdir(checkpoints_dir):
            if name.startswith('checkpoint-'):
                ask_and_delete(os.path.join(checkpoints_dir, name), is_dir=True)

    ask_and_delete('feedback.csv')
    clean_feedback_csv()

    ask_and_delete('news_data.csv')
    # Delete retrain_log.txt
    ask_and_delete('retrain_log.txt') 
