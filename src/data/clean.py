import json
import sys
import os
import ast
import pandas as pd
from src.data.preprocessing import CheXpertPreprocessing
from cleantext import clean

#The clean-text 0.6.0 package is used to clean the text fields. By default, it transliterates non-ASCII characters to 
# closest ASCII representation, fixes unicode sequences, lowercases the text, and removes line breaks.

#Utility Methods
def create_labels_dict(row):
    """Convert CheXpert labels to dictionary for a single row"""
    return {label: row[label] for label in CHEXPERT_LABELS if label in row}

def create_gt_list(findings, impression):
    gt = []
    output = ""
    if findings and findings != 'nan' and findings.lower() != 'none':
        output = output + "Findings: " + str(findings).replace("\n", "").strip() + " "
    if impression and impression != 'nan' and impression.lower() != 'none':
        output = output + "Impression: " + str(impression).replace("\n", "").strip()
    dict_findings = {"from": "gpt", "value": output}
    gt.append(dict_findings)
    return gt

def create_prompt_list(indication, examination):
    conversations = []
    if indication is not None and examination is not None:
        reason = indication.replace('\n', '')
        dict_reason = f"Provide a description of the findings and impression in the radiology image given the following indication: {reason}. The examination conducted was {examination}"
    elif indication is None and examination is not None:
        dict_reason = f"Provide a description of the findings and impression in the radiology image. The examination conducted was {examination}"
    elif indication is not None and examination is None:
        reason = indication.replace('\n', '')
        dict_reason = f"Provide a description of the findings and impression in the radiology image given the following indication: {reason}"
    else:
        dict_reason = "Provide a description of the findings and impression in the radiology image."
    conversations.append(dict_reason)
    return conversations

def clean_list_value(value):
    if isinstance(value, list):
        return ' '.join(map(str, value)).strip()
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return ' '.join(map(str, parsed)).strip()
            else:
                return str(parsed).strip()
        except (ValueError, SyntaxError):
            return value.strip("[]'\" ")
    return str(value).strip()

def clean_record(record):
    for k, v in record.items():
        if k in ["Examination", "Indication", "Impression"]:
            record[k] = clean_list_value(v)
        else:
            record[k] = str(v).strip()
    return record

#Set paths and constants
REQUIRED_COLUMNS = {'Examination', 'Indication', 'Findings', 'Impression'}
CHEXPERT_LABELS = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
    'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
    'Pleural Effusion','Pleural Other','Fracture','Support Devices']

def get_cleaned(config):
    #Load and process the input Chexpert data to fit the Llava-Rad format
    path = config.get("metadata", {})
    splits_path = config.get("splits", {})
    input_json_file = config.get("gt", {})

    chexpert_obj = CheXpertPreprocessing(path, splits_path)
    df = chexpert_obj.read_data()
    df['section_indication'] = (df['section_history'].fillna('') + " " + df['section_clinical_history'].fillna(''))
    df['chexpert_labels'] = df.apply(create_labels_dict, axis=1)

    # Printing out some useful metrics
    print("Number of null findings: ", df['section_findings'].isnull().sum())
    print("Number of null impressions: ", df['section_impression'].isnull().sum())
    print("Length of chexpert data after preprocessing: ", len(df))
    print(df['split'].value_counts())

    def chexpert_record(index):
        '''Method to process the input data file for the chexpert generate_method'''
        reason = clean(df.loc[index, 'section_indication'], no_line_breaks=True, lower = False)
        findings = clean(df.loc[index, 'section_findings'], no_line_breaks=True, lower = False)
        impressions = clean(df.loc[index, 'section_impression'], no_line_breaks=True, lower = False)
        examination = clean(df.loc[index, 'section_narrative'], no_line_breaks=True, lower = False)
        input_row = {
            'reason': reason,
            'findings': findings,
            'impressions': impressions,
            'examination': examination,
            'image': df.loc[index, 'path_to_image'],
            'generate_method': 'chexpert',
            'chexpert_labels': df.loc[index, 'chexpert_labels'],
            'split': df.loc[index, 'split'],
            'prompts': create_prompt_list(reason, examination),
            'gt': create_gt_list(findings, impressions)
        }
        return input_row

    def gpt_chexpert_record(index, record):
        new_record = {}
        new_record['reason'] = clean(record['Indication'], no_line_breaks=True, lower = False)
        new_record['findings'] = clean(record['Findings'], no_line_breaks=True, lower = False)
        new_record['impressions'] = clean(record['Impression'], no_line_breaks=True, lower = False)
        new_record['examination'] = clean(record['Examination'], no_line_breaks=True, lower = False)
        new_record['image'] = df.loc[index, 'path_to_image']
        new_record['generate_method'] = 'gpt'
        new_record['chexpert_labels'] = df.loc[index, 'chexpert_labels']
        new_record['split'] = df.loc[index, 'split']
        new_record['prompts'] = create_prompt_list(new_record['reason'], new_record['examination'])
        new_record['gt'] = create_gt_list(new_record['findings'], new_record['impressions'])
        return new_record

    records = []
    counter = -1
    with open(input_json_file, "r") as f:
        for line in f:
            counter += 1
            try:
                line = line.strip().replace('\\n', '').replace('\\', '')
                line = line.strip('"')
                record = json.loads(line)
                if isinstance(record, dict) and set(REQUIRED_COLUMNS) == set(record.keys()):
                    record = clean_record(record)
                    record = gpt_chexpert_record(counter, record)
                    records.append(record)
                else:
                    record = chexpert_record(counter)
                    records.append(record)
            except json.JSONDecodeError as e:
                record = chexpert_record(counter)
                records.append(record)
                continue

    input_json_df = pd.DataFrame(records)
    assert counter + 1 == len(input_json_df), "Mismatch in number of records processed"
    print("Generate_Method value counts in processed GPT data: ", input_json_df['generate_method'].value_counts())

    with open("gpt_processed_data.jsonl", "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return input_json_df