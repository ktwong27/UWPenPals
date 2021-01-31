# UWPenPals

UW Pen Pals is a social networking program created by Renee Diaz in August 2020. Since then, we've matched University of Washington students together as pen-pals to help build community amidst the pandemic. We distribute a Google Forms survey for students to sign up. This repository contains the code to match penpals together based on Google Forms responses and to send out mass emails with everybody's penpal information.

### Match-making Algorithm
In order to run the match maker, you need to create a json file with all the attributes you wish to compare. See more details in `Template/attributes_template.json.txt`. Once created, name it `attributes.json`

Go to your google form, click on responses and click the button to view responses in Sheets. In Google Sheets, navigate to `File->Download->Tab-separated Values (.tsv)`. Save it in the root directory and name it `responses.tsv`

To run the matchmaker:
```
python3 ./MatchMaker/matchmaker.py responses.tsv attributes.json
```

This will create two new files:
`ratings.tsv`
`penpal_matches.tsv`

`ratings.tsv` outputs the ratings given by the matchmaker. Each row represents each person in the Google Sheet (in order), where each column in that row represents the rating for matching the current row with the person in the row corresponding to the index of the colunn. Consider this example:
```
0   4.32    5.23
5.21    0   2.53
6.12    1.52    0
```
The first row has 0, 4.32, and 5.23 indicating that the person in the first row has a 0 rating to match with themselves (the first row), a 4.32 rating to match with the second row, and a 5.23 rating to match with the third row.

`penpal_matches.tsv` outputs the penpal matches themselves. This will be outputted in the following format:
```
Hash #  Name    Address Confirmed Address   Email   Confirmed Email Pronouns    Penpal 1    Penpal 2
```
Penpal 1 will contain the Hash # of the penpal for this person. Penpal 2 will contain the Hash # of the second penpal for this person, or -1 if they don't have a second penpal.

### Sending Emails
You must send out emails from a gmail account. In order to allow the `send_email` script to send emails on behalf of the address you provide, you must go to this link and turn ON the option to allow less secure apps: https://myaccount.google.com/lesssecureapps

From there, you can run the send_email script using the following:
```
python3 send_email.py email_body_txt data_tsv
```

`email_body_txt` must be a `.txt` file containing the body of the email you want to send out. It must contain the following strings: "{name}", "{penpal}", "{address}", and "{pronouns}" for python's `string.format()` to work correctly.

`data_tsv` must be a `.tsv` file containing the pen pal matches data in the following form:
```
Email	Name	Penpal's Name	Penpal's Address	Penpal's Pronouns
```

Once you run the script, you will be asked to input the gmail you wish to send the emails from, the google password for that email, and the subject line for the email. Once entered, the script will send an email for each line in `data_tsv`.

After running `send_email.py`, consider going back to this link to turn OFF the option to allow less secure apps: https://myaccount.google.com/lesssecureapps. This will restore Google's normal security settings to your email.