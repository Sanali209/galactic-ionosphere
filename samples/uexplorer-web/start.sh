#!/bin/bash
# Start script for UExplorer Web
# This script starts both the backend and frontend servers

set -e

echo "================================================"
echo "🚀 Starting UExplorer Web"
echo "================================================"
echo ""

# Check if we're in the correct directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the uexplorer-web directory"
    exit 1
fi

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python is not installed"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    exit 1
fi

echo "📦 Installing backend dependencies..."
cd backend
pip install -q -r requirements.txt
cd ..

echo "📦 Installing frontend dependencies..."
cd frontend
npm install --silent
cd ..

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "================================================"
echo "Starting servers..."
echo "================================================"
echo ""

# Start backend in background
echo "🔧 Starting FastAPI backend on http://localhost:8000"
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting Svelte frontend on http://localhost:5173"
cd frontend
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
