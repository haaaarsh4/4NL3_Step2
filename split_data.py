import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dialogue.csv').dropna(subset=['dialogue', 'name'])

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['name'])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['name'])

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)

test_df.drop(columns=['name']).to_csv('test.csv', index=False)

test_df.to_csv('test_correct.csv', index=False)

print(f"Train size: {len(train_df)}")
print(f"Val size:   {len(val_df)}")
print(f"Test size:  {len(test_df)}")