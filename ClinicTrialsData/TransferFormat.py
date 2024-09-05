import pandas as pd

df = pd.read_excel('20240810_Trial_Listing.xlsx')

df.to_csv('20240810_Trial_Listing.csv', index=False)