from flask import Flask

UPLOAD_FOLDER = '/Users/zzy9920/Desktop/first_app/uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
