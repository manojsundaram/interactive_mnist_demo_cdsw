from flask import Flask,send_from_directory,request,send_file
import logging
from IPython.display import Javascript,HTML

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app= Flask(__name__,static_url_path='')
    
@app.route('/')
def index():
    return "<script> window.location.href = '/flask/index.html'</script>"

@app.route('/flask/<path:path>')
def send_file(path):
    return send_from_directory('flask', path)


if __name__=="__main__":
  app.run(host="127.0.0.1", 
          port=int(os.environ['CDSW_APP_PORT']))
