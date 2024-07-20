import pandas as pd
# ignore warnings
import warnings
warnings.filterwarnings('ignore')

class ETL_Pipeline:
    """This class defines the ETL (Extract, Transform, Load) pipeline 
    for the email marketing campaign data. The class provides methods to clean and transform the raw data, 
    creating features that will be useful for implementing Q-learning."""
    def __init__(self):
        self.data = None
        self.user_base = None
        self.sent_emails = None
        self.responded = None
    def extract(self, data_path):
        """Load the raw data from the csv file."""
        data = pd.read_csv(data_path, low_memory=False)
        print("Data extracted successfully!")
        return data
    def transform(self):
        """Cleans the raw data and performs various preprocessing steps to prepare features."""
        # Drop duplicate rows between the sent_emails and responded tables
        if self.sent_emails.duplicated().any() or self.responded.duplicated().any():
            self.sent_emails.drop_duplicates(inplace=True)
            self.responded.drop_duplicates(inplace=True)
            print("Duplicate rows removed.")
        # Bin the numerical features
        custom_bins = {'Age': [0, 20, 30, 40, 50, 60, 70],
                       'Tenure': [0, 11, 14, 22, 38]}
        self.bin_custom_features(custom_bins)

        # Deal with NaN values in Tenure_binned
        self.user_base['Tenure_binned'].fillna(0, inplace=True)
        # Convert tenure_bin to integer values
        self.user_base['Tenure_binned'] = self.user_base['Tenure_binned'].astype(int)

        # Calculate the conversion rate
        conversion_df = self.calculate_conversion_rate()
        additional_features = ['Age_binned', 'Tenure_binned', 'Gender', 'Type', 'Customer_ID']  # SubjectLine_ID
        # additional_data = self.user_base[additional_features]
        additional_data = self.user_base[['Age_binned', 'Tenure_binned', 'Gender', 'Type', 'Customer_ID']]

        # Encode categorical features
        gender_encoding = {'M': 0, 'F': 1}
        type_encoding = {'B': 0, 'C': 1}
        additional_data['Gender'] = additional_data['Gender'].map(gender_encoding)
        additional_data['Type'] = additional_data['Type'].map(type_encoding)

        # Merge the data
        self.data = pd.merge(conversion_df, additional_data, on='Customer_ID', how='left')
        self.determine_action()
    def load(self):
        """Loads the transformed data to a CSV file."""
        if self.data is not None:
            self.data.to_csv('data/transformed_data.csv', index=False)
            print("Data loaded successfully!")
        else:
            print("No data available to load. Please run the transform method first.")
    #### Helper functions ####
    def bin_custom_features(self, custom_bins):
        """Bins the custom features according to the specified bins."""
        if self.user_base is not None:
            for feature, bins in custom_bins.items():
                if feature in self.user_base.columns and pd.api.types.is_numeric_dtype(self.user_base[feature]):
                    bins = [int(bin) for bin in bins]
                    self.user_base[f'{feature}_binned'] = pd.cut(self.user_base[feature], bins=bins, labels=False)
                    print(f"{feature} binned successfully.")
                else:
                    print(f"{feature} is not a valid numerical feature.")
        else:
            print("No user base data available.")
    def calculate_conversion_rate(self):
        merged_df = pd.merge(self.user_base, self.sent_emails, on='Customer_ID')
        merged_df = pd.merge(merged_df, self.responded, on=['Customer_ID', 'SubjectLine_ID'], how='left')

        selected_data = merged_df[['Customer_ID', 'Sent_Date', 'Responded_Date', 'SubjectLine_ID']]
        sent_counts = selected_data.groupby('Customer_ID').size().reset_index(name='Emails_Sent')
        responded_counts = selected_data.groupby('Customer_ID')['Responded_Date'].count().reset_index(name='Responses')
        conversion_df = pd.merge(sent_counts, responded_counts, on='Customer_ID', how='left')

        # Calculate the conversion rate
        conversion_df['Conversion_Rate'] = round((conversion_df['Responses'] / conversion_df['Emails_Sent']) * 100, 2)
        # Add conversion rate to data and drop unnecessary columns
        final_df = pd.merge(selected_data.drop(columns=['Sent_Date', 'Responded_Date']), conversion_df, on='Customer_ID', how='inner')
        final_df = final_df.drop_duplicates()
        return final_df
    def determine_action(self):
        actions = []
        for _, row in self.data.iterrows():
            subject_line = row['SubjectLine_ID']

            # Determine the action based on the subject line ID
            if subject_line == 1:
                action = 1  # Send email with Subject Line 1
            elif subject_line == 2:
                action = 2  # Send email with Subject Line 2
            elif subject_line == 3:
                action = 3  # Send email with Subject Line 3
            else:
                action = 4  # Do not send email
            
            actions.append(action)
        
        self.data['Action'] = actions
        print("Actions determined successfully.")

if __name__ == "__main__":
    # Initialize the ETL pipeline
    etl = ETL_Pipeline()

    # Extract data
    etl.user_base = etl.extract('data/userbase.csv')
    etl.sent_emails = etl.extract('data/sent_emails.csv')
    etl.responded = etl.extract('data/responded.csv')

    # Transform data
    etl.transform()

    # Load data
    etl.load()

    # Transformed data should now be saved to the data directory as 'transformed_data.csv'