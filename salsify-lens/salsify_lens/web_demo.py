from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Define a mapping of product names to their respective iframe URLs
PRODUCT_URLS = {
    'coca_cola': 'https://salsify-ecdn.com/sdk/s-79dc3f14-7e41-440b-ac08-a0cae4ae83bc/en-US/BTF/GTIN/00049000000443/index.html',
    'pepsi': 'https://www.example.com/pepsi_product',
    'sprite': 'https://www.example.com/sprite_product',
    # Add more products as needed
}

# Initialize a variable to store the current product
current_product = None

# Define the HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Viewer</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            height: 100vh;
            background-color: #f0f0f0;
            margin: 0;
        }
        h1 {
            margin-bottom: 20px;
        }
        iframe {
            width: 80%;
            height: 80%;
            border: none;
        }
    </style>
    <script>
        let currentProduct = "{{ product_name }}"; // Initialize with the server-provided product name

        function checkForProductUpdate() {
            fetch('/current_product')
                .then(response => response.json())
                .then(data => {
                    console.log('Current product:', data.product);
                    console.log('Previous product:', currentProduct);
                    if (data.product !== currentProduct) {
                        currentProduct = data.product;
                        window.location.href = '/?product=' + encodeURIComponent(currentProduct); // Redirect to new product URL
                    }
                })
                .catch(err => console.error('Error fetching product:', err));
        }

        // Check for product updates every 1 seconds
        setInterval(checkForProductUpdate, 1000);
    </script>
</head>
<body>
    <h1>{{ product_name.title().replace('_', '-') }} product found:</h1>
    {% if iframe_url %}
        <iframe src="{{ iframe_url }}" title="{{ product_name }} product iframe"></iframe>
    {% else %}
        <p>Search for a product!</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    global current_product
    product_name = request.args.get('product', '').lower()
    
    if product_name:
        current_product = product_name
        iframe_url = PRODUCT_URLS.get(product_name)
    else:
        iframe_url = None
    
    print(">>> product_name: ", product_name)
    print(">>> iframe_url: ", iframe_url)

    return render_template_string(HTML_TEMPLATE, product_name=product_name, iframe_url=iframe_url)

@app.route('/current_product')
def current_product_endpoint():
    return jsonify({'product': current_product})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8042)
