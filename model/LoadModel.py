from openai import OpenAI
import os
import time
# set API key
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)
import pandas as pd

import asyncio
from concurrent.futures import ThreadPoolExecutor

# load model
# GPT-3: "gpt-3.5-turbo"; davinci："text-davinci-003"
class GPT3Model:
    def __init__(self, api_key):
        OpenAI.api_key = key

    def generate_summary(self, description, max_tokens=150, temperature=0.7):
        retries = 5  # max request
        for i in range(retries):
            try:
                print(f"Attempt {i + 1}: Generating summary for description...")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant for summarizing clinical trial descriptions."},
                        {"role": "user",
                         "content": f"Summarize the following clinical trial description: {description}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # summary
                print(f"Summary successfully generated on attempt {i + 1}")
                return response.choices[0].message.content.strip()

            except OpenAI.error.RateLimitError:
                # request rate
                wait_time = 2 ** i  # time wait
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"An error occurred on attempt {i + 1}: {e}")
                break

        print("Failed to generate summary after maximum retries.")
        return None



if __name__ == "__main__":
  # load model
  gpt3 = GPT3Model(api_key=key)

  # load data file
  df = pd.read_csv('../ClinicTrialsData/chosen_data.csv', encoding='ISO-8859-1')

  data_to_save = []

  # summary
  for index, row in df.iterrows():
    trial_name = row['Trial Name']
    trial_id = row['Trial ID']
    description = row['A short description of the trial']

    summary = gpt3.generate_summary(description)

    # save
    data_to_save.append({
      'Trial Name': trial_name,
      'Trial ID': trial_id,
      'A short description of the trial': description,
      'Generated Summary': summary
    })

  df_summary = pd.DataFrame(data_to_save)
  df_summary.to_csv('../ClinicTrialsData/summary_data.csv', index=False, encoding='utf-8')

