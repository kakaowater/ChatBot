# -*- coding: utf-8 -*-

from app.main import app
from flask_ngrok import run_with_ngrok

if __name__ == "__main__":
    # app.run(port=6006, debug=True)
    run_with_ngrok(app)
    app.run()
