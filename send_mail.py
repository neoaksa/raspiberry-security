import smtplib

def send_mail(subject, message):
    SERVER = "smtp.live.com"
    FROM = "raspiberrypi@hotmail.com"
    pwd = 'iamaksa333'
    TO = ["neo_aksa@hotmail.com"] # must be a list

    SUBJECT = subject
    TEXT = message

    # Prepare actual message
    message = """From: %s\r\nTo: %s\r\nSubject: %s\r\n\

    %s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)

    # Send the mail

    server = smtplib.SMTP(SERVER,587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(FROM, pwd)
    server.sendmail(FROM, TO, message)
    server.quit()

if __name__ == '__main__':
    send_mail('test','this is a test message!')