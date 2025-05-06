import pandas as pd
import numpy as np
import os
import kagglehub 

# --- 1. Download the dataset ---
print("Downloading SMS Spam Collection Dataset...")
try:
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    print(f"Dataset downloaded to: {path}")

    # --- 2. Find the CSV file ---
    file_path = None
    csv_filename = 'spam.csv'
    print(f"Looking for '{csv_filename}' in downloaded directory...")
    for root, dirs, files in os.walk(path):
        if csv_filename in files:
            file_path = os.path.join(root, csv_filename)
            break

    df = pd.DataFrame() # Initialize an empty DataFrame

    if file_path:
        print(f"Found dataset file at: {file_path}")
        # --- 3. Load the dataset ---
        try:
            # This dataset is often tab-separated, and might not have headers by default
            # Let's try reading with common options or assume it's simple csv
            # Based on typical structure, it's two columns v1 (label) and v2 (text)
            # Let's try reading as CSV first, if that fails, try tab-separated
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
                # Check if columns look like standard v1, v2
                if 'v1' in df.columns and 'v2' in df.columns:
                     df = df[['v1', 'v2']] # Select only relevant columns
                     df.columns = ['Label', 'Message'] # Rename columns for clarity
                     print("Dataset loaded successfully with columns 'Label' and 'Message'.")
                else:
                     # Try reading as tab-separated if CSV didn't yield expected columns
                     print("CSV read didn't yield expected columns. Trying tab separation...")
                     df = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'Message'], encoding='latin-1')
                     print("Dataset loaded successfully with columns 'Label' and 'Message' using tab separation.")

            except Exception as e_read:
                 print(f"Error reading CSV/TSV file: {e_read}")


            # --- Initial Data Prep (if loading was successful) ---
            if not df.empty:
                # Add a message length column for analysis
                df['Message_Length'] = df['Message'].str.len()
                print("Added 'Message_Length' column.")

                # Add a word count column (simple split by space)
                df['Word_Count'] = df['Message'].str.split().str.len()
                print("Added 'Word_Count' column.")

                print("\nDataset Info:")
                df.info()
                print("\nDataset Head:")
                print(df.head())

        except Exception as e_load:
            print(f"An error occurred while loading or processing the dataset: {e_load}")
            df = pd.DataFrame() # Ensure df is empty if loading fails

    else:
        print(f"Error: '{csv_filename}' not found in the downloaded directory or its subdirectories.")
        df = pd.DataFrame() # Ensure df is empty if file not found

except Exception as e_download:
    print(f"An error occurred during dataset download: {e_download}")
    df = pd.DataFrame() # Ensure df is empty if download fails


# --- Perform Data Analysis Tasks (20 Problem Statements) ---

if not df.empty:
    print("\n--- Performing Data Analysis Tasks ---")

    # Problem Statement 1: What is the total number of messages in the dataset?
    total_messages = len(df)
    print(f"\n1. Total number of messages: {total_messages}")

    # Problem Statement 2: How many messages are labeled as 'spam'?
    spam_count = df[df['Label'] == 'spam'].shape[0]
    print(f"\n2. Number of spam messages: {spam_count}")

    # Problem Statement 3: How many messages are labeled as 'ham'?
    ham_count = df[df['Label'] == 'ham'].shape[0]
    print(f"\n3. Number of ham messages: {ham_count}")

    # Problem Statement 4: What is the proportion of spam messages in the dataset?
    proportion_spam = spam_count / total_messages if total_messages > 0 else 0
    print(f"\n4. Proportion of spam messages: {proportion_spam:.2%}")

    # Problem Statement 5: What is the proportion of ham messages in the dataset?
    proportion_ham = ham_count / total_messages if total_messages > 0 else 0
    print(f"\n5. Proportion of ham messages: {proportion_ham:.2%}")

    # Problem Statement 6: Display the column names of the DataFrame.
    column_names = df.columns.tolist()
    print(f"\n6. Column names: {column_names}")

    # Problem Statement 7: Find the message with the longest character length.
    if 'Message_Length' in df.columns and 'Message' in df.columns:
        longest_message_index = df['Message_Length'].idxmax()
        longest_message = df.loc[longest_message_index]
        print(f"\n7. Message with the longest character length ({longest_message['Message_Length']} characters):")
        print(f"   Label: {longest_message['Label']}")
        print(f"   Message: {longest_message['Message'][:100]}...") # Print first 100 chars
    else:
         print("\n7. Required column(s) ('Message_Length' or 'Message') not found.")


    # Problem Statement 8: Find the message with the shortest character length (excluding empty messages).
    if 'Message_Length' in df.columns and 'Message' in df.columns:
        shortest_message_index = df[df['Message_Length'] > 0]['Message_Length'].idxmin()
        shortest_message = df.loc[shortest_message_index]
        print(f"\n8. Message with the shortest character length ({shortest_message['Message_Length']} characters):")
        print(f"   Label: {shortest_message['Label']}")
        print(f"   Message: {shortest_message['Message']}")
    else:
         print("\n8. Required column(s) ('Message_Length' or 'Message') not found.")


    # Problem Statement 9: Calculate the average character length of all messages.
    if 'Message_Length' in df.columns:
        average_length = df['Message_Length'].mean()
        print(f"\n9. Average message character length: {average_length:.2f}")
    else:
        print("\n9. 'Message_Length' column not found.")


    # Problem Statement 10: Calculate the average character length of spam messages.
    if 'Message_Length' in df.columns and 'Label' in df.columns:
        spam_messages = df[df['Label'] == 'spam']
        if not spam_messages.empty:
            average_length_spam = spam_messages['Message_Length'].mean()
            print(f"\n10. Average character length of spam messages: {average_length_spam:.2f}")
        else:
            print("\n10. No spam messages found.")
    elif not df.empty:
        print("\n10. Required column(s) ('Message_Length' or 'Label') not found.")


    # Problem Statement 11: Calculate the average character length of ham messages.
    if 'Message_Length' in df.columns and 'Label' in df.columns:
        ham_messages = df[df['Label'] == 'ham']
        if not ham_messages.empty:
            average_length_ham = ham_messages['Message_Length'].mean()
            print(f"\n11. Average character length of ham messages: {average_length_ham:.2f}")
        else:
            print("\n11. No ham messages found.")
    elif not df.empty:
         print("\n11. Required column(s) ('Message_Length' or 'Label') not found.")


    # Problem Statement 12: Find and display the first 5 messages containing the word 'free'.
    if 'Message' in df.columns:
        keyword = 'free'
        messages_with_keyword = df[df['Message'].str.contains(keyword, case=False, na=False)].head()
        print(f"\n12. First 5 messages containing '{keyword}':")
        if not messages_with_keyword.empty:
            for index, row in messages_with_keyword.iterrows():
                print(f"   - Label: {row['Label']}, Message: {row['Message'][:80]}...") # Print first 80 chars
        else:
            print(f"   No messages found containing '{keyword}'.")
    else:
        print("\n12. 'Message' column not found.")


    # Problem Statement 13: Count how many messages contain the word 'win'.
    if 'Message' in df.columns:
        keyword = 'win'
        count_with_keyword = df['Message'].str.contains(keyword, case=False, na=False).sum()
        print(f"\n13. Number of messages containing '{keyword}': {count_with_keyword}")
    else:
        print("\n13. 'Message' column not found.")


    # Problem Statement 14: Display the value counts for the 'Label' column.
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print(f"\n14. Value counts for 'Label' column:")
        print(label_counts)
    else:
        print("\n14. 'Label' column not found.")


    # Problem Statement 15: Find the message with the most words.
    if 'Word_Count' in df.columns and 'Message' in df.columns:
        longest_word_count_index = df['Word_Count'].idxmax()
        message_most_words = df.loc[longest_word_count_index]
        print(f"\n15. Message with the most words ({message_most_words['Word_Count']} words):")
        print(f"   Label: {message_most_words['Label']}")
        print(f"   Message: {message_most_words['Message'][:100]}...") # Print first 100 chars
    else:
        print("\n15. Required column(s) ('Word_Count' or 'Message') not found.")


    # Problem Statement 16: Display the first 10 spam messages.
    if 'Label' in df.columns and 'Message' in df.columns:
        first_10_spam = df[df['Label'] == 'spam'].head(10)
        print(f"\n16. First 10 spam messages:")
        if not first_10_spam.empty:
             for index, row in first_10_spam.iterrows():
                print(f"   - {row['Message'][:100]}...") # Print first 100 chars
        else:
            print("   No spam messages found.")
    elif not df.empty:
        print("\n16. Required column(s) ('Label' or 'Message') not found.")


    # Problem Statement 17: Display the first 10 ham messages.
    if 'Label' in df.columns and 'Message' in df.columns:
        first_10_ham = df[df['Label'] == 'ham'].head(10)
        print(f"\n17. First 10 ham messages:")
        if not first_10_ham.empty:
            for index, row in first_10_ham.iterrows():
                print(f"   - {row['Message'][:100]}...") # Print first 100 chars
        else:
            print("   No ham messages found.")
    elif not df.empty:
        print("\n17. Required column(s) ('Label' or 'Message') not found.")


    # Problem Statement 18: Calculate the standard deviation of message length.
    if 'Message_Length' in df.columns:
        std_dev_length = df['Message_Length'].std()
        print(f"\n18. Standard deviation of message character length: {std_dev_length:.2f}")
    else:
        print("\n18. 'Message_Length' column not found.")


    # Problem Statement 19: Calculate the average word count for spam messages.
    if 'Word_Count' in df.columns and 'Label' in df.columns:
         spam_messages = df[df['Label'] == 'spam']
         if not spam_messages.empty:
             average_word_count_spam = spam_messages['Word_Count'].mean()
             print(f"\n19. Average word count for spam messages: {average_word_count_spam:.2f}")
         else:
             print("\n19. No spam messages found.")
    elif not df.empty:
        print("\n19. Required column(s) ('Word_Count' or 'Label') not found.")


    # Problem Statement 20: Count messages where the character length is greater than 200.
    if 'Message_Length' in df.columns:
        long_messages_count = df[df['Message_Length'] > 200].shape[0]
        print(f"\n20. Number of messages with character length > 200: {long_messages_count}")
    else:
        print("\n20. 'Message_Length' column not found.")

else:
    print("\nDataFrame is empty. Cannot perform analysis tasks.")