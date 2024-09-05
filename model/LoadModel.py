import openai
from openai import AsyncOpenAI
import os
import time
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

# set API key and initialize Async client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# Define async GPT-3 model class
class GPT3Model:
    def __init__(self, client):
        self.client = client

    async def generate_summary(self, description, max_tokens=150, temperature=0.7):
        retries = 5  # max request
        for i in range(retries):
            start_time = time.time()  # 开始计时
            try:
                print(f"Attempt {i + 1}: Generating summary for description...")
                response = await self.client.chat.completions.create(
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
                end_time = time.time()  # 结束计时
                elapsed_time = end_time - start_time  # 计算总耗时
                # summary
                print(f"Summary successfully generated on attempt {i + 1} (Time taken: {elapsed_time:.2f} seconds)")
                return response.choices[0].message.content.strip()

            except client.error.RateLimitError:
                # request rate
                wait_time = 2 ** i  # time wait
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

            except Exception as e:
                print(f"An error occurred on attempt {i + 1}: {e}")
                break

        print("Failed to generate summary after maximum retries.")
        return None


# Main async function
async def main():
    # load model
    gpt3 = GPT3Model(client)

    # load data file
    df = pd.read_csv('../ClinicTrialsData/chosen_data.csv', encoding='ISO-8859-1')

    data_to_save = []

    # summary
    for index, row in df.iterrows():
        trial_name = row['Trial Name']
        trial_id = row['Trial ID']
        description = row['A short description of the trial']

        try:
            summary = await gpt3.generate_summary(description)

            # save
            data_to_save.append({
                'Trial Name': trial_name,
                'Trial ID': trial_id,
                'A short description of the trial': description,
                'Generated Summary': summary
            })

        except Exception as e:
            # 捕捉并报告错误，但继续处理下一条数据
            print(f"An error occurred while processing trial {trial_id}: {e}")
            continue

    # Convert to DataFrame and attempt to save the data
    try:
        df_summary = pd.DataFrame(data_to_save)
        df_summary.to_csv('../ClinicTrialsData/summary_data.csv', index=False, encoding='utf-8')
        print("Data successfully saved to summary_data.csv.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")


# Entry point for the async execution
if __name__ == "__main__":
    asyncio.run(main())

