import csv
import plotly.express as px
from flask import Flask, jsonify
import argparse
from datetime import datetime

def parseargs():
    parser = argparse.ArgumentParser(description='Plot data from CSV file.')
    parser.add_argument("-i", "--input_file", type=str, help="Input CSV file")
    return parser.parse_args()

args = parseargs()

app = Flask(__name__)

x = []
y = []

with open(args.input_file) as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))  # Convert string to datetime object
        # Handle empty strings in the data
        if row[1] != '':
            y.append(float(row[1]))
        else:
            y.append(None)  # or use 0.0 if that's more appropriate

fig = px.line(x=x, y=y)

@app.route('/')
def index():
    return '''
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div id='plot'></div>
                <script>
                   var graph = ''' + fig.to_json() + ''';
                   Plotly.plot('plot',graph);
                </script>
            </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=6006)

