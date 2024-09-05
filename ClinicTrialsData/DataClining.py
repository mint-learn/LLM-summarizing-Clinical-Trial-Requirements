import pandas as pd
import re

# load data
df = pd.read_csv('20240810_Trial_Listing.csv', encoding='ISO-8859-1')

# 1. delete 'NA'
df.replace('NA', pd.NA, inplace=True)
df_filtered = df.dropna(subset=['A short description of the trial', 'Participant criteria that the trial is looking for'])

df.dropna(inplace=True)

# 2. standarization
columns_to_strip = ['Trial Name', 'Trial short-form', 'Principal Investigator',
                    'Disease(s)/Condition(s)', 'A short description of the trial',
                    'Participant criteria that the trial is looking for']
for column in columns_to_strip:
    df[column] = df[column].str.strip().str.lower()


# 3. special characters
def clean_special_chars(text):
    if pd.isna(text):
        return text

    # '/' to 'or'
    text = text.replace('/', ' or ')

    # remaining：+、-、.、,、:、;、()
    text = re.sub(r'[^\w\s\-\+.,:;()]', '', text)

    # space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


columns_to_clean = ['A short description of the trial', 'Participant criteria that the trial is looking for']
for column in columns_to_clean:
    df[column] = df[column].apply(clean_special_chars)

# 4. save to file
df_filtered.to_csv('2ColumnsFiltered.csv', index=False)
df.to_csv('chosen_data.csv', index=False)

