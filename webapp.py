from flask import Flask, render_template

# Import your functions from the program
from main import question3, question4, question5, question6

app = Flask(__name__)

# Define routes for each question
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/question3')
def predict_defect_length():
    # Call the function for question 3
    rmse_value = question3()  # Assuming question3() returns the RMSE value
    return render_template('question3.html', rmse=rmse_value)


@app.route('/question4')
def predict_net_weight():
    # Call the function for question 4
    question4()
    return render_template('question4.html')

@app.route('/question5')
def predict_defect_count():
    # Call the function for question 5
    question5()
    return render_template('question5.html')

@app.route('/question6')
def find_associations():
    # Call the function for question 6
    question6()
    return render_template('question6.html')

if __name__ == '__main__':
    app.run(debug=True)
