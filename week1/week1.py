import mailbox, csv
# Specify the path to your mbox file
mbox_file = 'mails.mbox'

# Create a mailbox object
mbox = mailbox.mbox(mbox_file)

# Define the email address of the sender you want to filter
sender_email = 'Associate Dean Student Affairs <adean_sa@iiitvadodara.ac.in>'

# Loop through the emails and print the body of emails from the specified sender
# Function to extract text from email message parts
output_csv_file = 'it_up_mails.csv'

def extract_text(message):
    if message.is_multipart():
        text = []
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                text.append(part.get_payload())
        return "\n".join(text)
    else:
        return message.get_payload()

# Create or open the CSV file for writing
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write headers to the CSV file
    csv_writer.writerow(['Subject', 'From', 'Body'])
    
    # Loop through the emails and write them to the CSV file
    for message in mbox:
        if message['From'] == sender_email:
            subject = message['Subject']
            sender = message['From']
            body = extract_text(message)
            
            # Write the email content to the CSV file
            csv_writer.writerow([subject, sender, body])