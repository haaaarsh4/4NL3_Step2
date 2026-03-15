import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dialogue.csv')

labels = df.columns[0]

counts = df[labels].value_counts()

counts.plot(kind='bar')

plt.title('Distribution of ground truth labels')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.xticks(rotation=45) 
plt.tight_layout() 

plt.savefig("distribution.png")