from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd

# Set up a headless Chrome browser
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

# Load the CSV file

csv_file_path = '/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/Data/Data_UserID/hate_speech_dataset.csv' # select dataset
df = pd.read_csv(csv_file_path)


############### for benevolent_sexist and hostile_sexist #############################
# Rename the first column to 'TweetID'
#if df.columns[0] != 'TweetID':
#   df.rename(columns={df.columns[0]: 'TweetID'}, inplace=True)

############### for benevolent_sexist and hostile_sexist #############################


############### for NAACL_SRW_2016 #####################
# Rename the first and second columns
#df.columns.values[0] = 'TweetID'
#df.columns.values[1] = 'annotation'
############### for NAACL_SRW_2016 #####################



# Check if the 'User' column exists, and if not, create it
if 'User' not in df.columns:
    df['User'] = ""

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Check if the 'User' column is empty for this row
    if pd.isna(row['User']) or not row['User']:
        #tweet_id = row['tweet id'] # for twitter_data_waseem_hovy
        tweet_id = row['TweetID'] # for the rest
        try:
            # Construct the URL with the tweet ID
            url = f"https://twitter.com/user/status/{tweet_id}"

            # Navigate to the URL
            driver.get(url)

            # Wait for the final URL to load and any redirects to complete
            wait = WebDriverWait(driver, 20)
            final_url = wait.until(EC.url_changes(url))

            # Get the current URL after the wait
            final_url = driver.current_url

            # Extract the user by splitting the final URL
            url_parts = final_url.split("/")
            user = url_parts[3]

            # Set the user in the DataFrame
            df.at[index, 'User'] = user

            # Print the final URL and extracted user
            print("Tweet ID:", tweet_id)
            print("Final URL:", final_url)
            print("User:", user)

            # Save the DataFrame to the CSV file after updating the 'User' column
            df.to_csv(csv_file_path, index=False)

        except TimeoutException:
            print(f"User not found for Tweet ID {tweet_id}")
            df.at[index, 'User'] = "Not found"
            # Save the DataFrame to the CSV file after updating the 'User' column
            df.to_csv(csv_file_path, index=False)

# Close the browser
driver.quit()
