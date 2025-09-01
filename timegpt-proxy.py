#!/usr/bin/env python3
"""
TimeGPT Proxy Server - Solves CORS restrictions
Run: python timegpt-proxy.py
Then update frontend to call localhost:8000 instead of api.nixtla.io
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Allow all origins

TIMEGPT_API_KEY = "nixak-NsKC5AcRal1bByr7Bp3JzJEpd8hS0r8X1GYoElLZww5smMtKfCyPISaE8oR8DZ7nqvTG2y93NmeEo1Jl"

@app.route('/timegpt', methods=['POST'])
def proxy_timegpt():
    """Proxy requests to TimeGPT API"""
    try:
        # Forward the request to TimeGPT
        response = requests.post(
            'https://api.nixtla.io/timegpt',
            headers={
                'Authorization': f'Bearer {TIMEGPT_API_KEY}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            json=request.json,
            timeout=30
        )
        
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'TimeGPT Proxy'})

if __name__ == '__main__':
    print("🚀 Starting TimeGPT Proxy Server...")
    print("📡 Frontend should call: http://localhost:8000/timegpt")
    print("🔑 Using TimeGPT API key:", TIMEGPT_API_KEY[:15] + "...")
    app.run(host='0.0.0.0', port=8000, debug=True)