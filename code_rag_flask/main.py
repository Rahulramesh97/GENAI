from flask import Flask
from app.routes import configure_routes
import os

# Ensure Flask knows where to find templates
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "app/templates"))

# Pass the app instance, NOT a string
configure_routes(app)

if __name__ == "__main__":
    app.run(debug=True)  # Change to `False` in production
 