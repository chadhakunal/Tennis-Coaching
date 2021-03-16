from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

@app.route('/')
def index():
   return render_template('index.html')
	
# @app.route('/get-coordinates', methods = ['POST'])
# def upload_file():
#    f = request.files['file']
#    image = f.stream
#    print(image)
#    # f.save(secure_filename(f.filename))
#    return '<h2>file uploaded successfully</h2>'

@socketio.on('connect', namespace='/')
def connect_web():
   print('[INFO] Web client connected: {}'.format(request.sid))
   
@socketio.on('disconnect', namespace='/')
def disconnect_web():
   print('[INFO] Web client disconnected: {}'.format(request.sid))


@socketio.on('get-coordinates', namespace='/')
def get_coordinates(data):
   print('[INFO] Web client Sent Data: {}'.format(request.sid))


if __name__ == '__main__':
   print('[INFO] Starting server at http://localhost:5000')
   socketio.run(app=app, host='0.0.0.0', port=5000)