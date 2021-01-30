# UWPenPals

UW Pen Pals is a social networking program created by Renee Diaz in August 2020. Since then, we've matched University of Washington students together as pen-pals to help build community amidst the pandemic. We distribute a Google Forms survey for students to sign up. This repository contains the code to match penpals together based on Google Forms responses and to send out mass emails with everybody's penpal information.

### Match-making Algorithm

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