import pandas as pd

# Load the CSV file
file_path = 'filtered_spam_data_utf8.csv'
data = pd.read_csv(file_path)

# Filter the data to keep only rows where SpamLabel = 0
filtered_data = data[data['SpamLabel'] == 0]

# Define the output file path
output_file_path = 'spam_data.csv'

# Save the filtered data to a new CSV file, ensuring UTF-8 encoding
filtered_data.to_csv(output_file_path, index=False, encoding='utf-8')

