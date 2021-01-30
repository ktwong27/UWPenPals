#imports
import smtplib
import traceback
import sys
import csv

"""
Usage: python3 send_email.py email_body_txt data_tsv

Parameters:
email_body_txt must be a .txt file containing the body of the email you want to send out. It must contain the following strings: "{name}", "{penpal}", "{address}", and "{pronouns}" for python's string.format() to work correctly.

data_tsv must be a .tsv file containing the pen pal matches data in the following form:
Email	Name	Penpal's Name	Penpal's Address	Penpal's Pronouns
"""
def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    email_body_file = open(sys.argv[1], "r")

    gmail_user = input('What gmail are you sending from?\n')
    gmail_password = input('Enter password:\n')
    subject = input('Email subject:\n')

    sent_from = gmail_user
    data = list(csv.reader(open(sys.argv[2], "r"), delimiter="\t"))    
    body_template = email_body_file.read()

    for i in range(len(data)):
        send_to = data[i][0]
        body = body_template.format(name=data[i][1], penpal=data[i][2], address=data[i][3], pronouns=data[i][4])
        email_text = "From: {}\nTo: {}\nSubject: {}\n\n{}".format(sent_from, send_to, subject, body)

        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(gmail_user, gmail_password)
            server.sendmail(sent_from, send_to, email_text)
            server.close()

            print('Email', i + 1, 'sent!')
        except:
            traceback.print_exc()
            print('Something went wrong...')

if __name__ == "__main__":
    main()
