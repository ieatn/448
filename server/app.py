# create flask server
# import depression csv data 
# use linear regression and train the model on the data to predict percentage of depression based on survey
# api route file send survey input and call this server for the prediction
print('hello world')

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

