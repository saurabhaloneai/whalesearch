from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('frontend/index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q')
    # Here, you would perform the search logic based on the query
    # and get the search results
    results = [
        {'title': 'Example Result 1', 'snippet': 'This is an example search result.', 'url': 'http://example.com/1'},
        {'title': 'Example Result 2', 'snippet': 'Another example search result.', 'url': 'http://example.com/2'},
    ]
    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)