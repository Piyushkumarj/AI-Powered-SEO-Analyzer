from flask import Flask, render_template, request, jsonify
from main import SEOAnalyzer  # Import your SEOAnalyzer class

app = Flask(__name__)
analyzer = SEOAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form.get('url')  # Get URL from form
    if not url:
        return jsonify({'error': 'Please provide a URL'}), 400
    
    # Run SEO analysis
    results = analyzer.analyze_seo(url)
    if not results:
        return jsonify({'error': 'Could not analyze the website. Please check the URL and try again.'}), 500
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
