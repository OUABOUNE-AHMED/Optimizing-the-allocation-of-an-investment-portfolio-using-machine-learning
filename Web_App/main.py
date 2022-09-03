from flask import Blueprint, render_template, flash
from flask_login import login_required, current_user
from dash_application import create_dash_application
from __init__ import create_app, db



# our main blueprint
main = Blueprint('main', __name__)


@main.route('/')  # home page that return 'index'
def index():
    try :
        user_name = current_user.name
    except:
        user_name = ''
        pass
    return render_template('index.html', name=user_name)

#added this commnet to test if Git works 
@main.route('/dashboard')  # profile page that return 'profile'
@login_required
def dashboard():
    try :
        user_name = current_user.name
    except:
        user_name = ''
        pass
    return render_template('profile.html', name=user_name)


app = create_app()  # we initialize our flask app using the __init__.py function

create_dash_application(app)

if __name__ == '__main__':
    db.create_all(app=create_app())  # create the SQLite database
    app.run(debug=True)  # run the flask app on debug mode
