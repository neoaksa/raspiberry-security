import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_mail(subject, message,frame):
    SERVER = "smtp.live.com"
    FROM = "raspiberrypi@hotmail.com"
    pwd = 'iamaksa333'
    TO = "neo_aksa@hotmail.com" # must be a list

    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    msgRoot['From'] = FROM
    msgRoot['To'] = TO
    msgRoot.preamble = 'This is a multi-part message in MIME format.'

    # Encapsulate the plain and HTML versions of the message body in an
    # 'alternative' part, so message agents can decide which they want to display.
    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)

    # Prepare actual message
    msgText = MIMEText(message)
    msgAlternative.attach(msgText)

    # We reference the image in the IMG SRC attribute by the ID we give it below
    msgText = MIMEText('<b>some security issue happening</b><br><img src="cid:image1"><br>', 'html')
    msgAlternative.attach(msgText)
    # Load the frame
    msgImage = MIMEImage(frame,'jpg')
    msgImage.add_header('Content-ID', '<image1>')
    msgImage.add_header('Content-Disposition', 'inline', filename='screenshot')
    msgRoot.attach(msgImage)

    # Send the mail

    server = smtplib.SMTP(SERVER,587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(FROM, pwd)
    server.sendmail(FROM, TO, msgRoot.as_string())
    server.quit()

if __name__ == '__main__':
    fp = open('/home/pi/code/raspiberry-security/cam.jpg','rb')
    send_mail('test','this is a test message!',fp.read())
    fp.close()