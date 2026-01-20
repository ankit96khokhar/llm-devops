#!/bin/bash

echo "🦙 Setting up LOCAL LLM options for Kubernetes Incident Response"
echo "Choose your preferred LLM setup:"
echo ""
echo "1. 🆓 Rule-based (No LLM) - Completely free, instant"
echo "2. 🦙 Ollama + Llama - Best balance, easy setup"  
echo "3. 🤗 Hugging Face - More options, needs GPU/RAM"
echo ""

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "✅ Using rule-based analysis (no additional setup needed)"
        export LLM_TYPE="rules"
        ;;
    2)
        echo "🦙 Setting up Ollama for local Llama models..."
        
        # Install Ollama
        if ! command -v ollama &> /dev/null; then
            echo "📦 Installing Ollama..."
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                curl -fsSL https://ollama.ai/install.sh | sh
            else
                # Linux
                curl -fsSL https://ollama.ai/install.sh | sh
            fi
        else
            echo "✅ Ollama already installed"
        fi
        
        # Start Ollama server
        echo "🚀 Starting Ollama server..."
        ollama serve &
        sleep 5
        
        # Pull recommended model
        echo "📥 Downloading Llama 3.2 (3B model - good for incident analysis)..."
        ollama pull llama3.2:latest
        
        echo "✅ Ollama setup complete!"
        echo "Available models:"
        ollama list
        
        export LLM_TYPE="ollama"
        export OLLAMA_HOST="http://localhost:11434"
        ;;
    3)
        echo "🤗 Setting up Hugging Face transformers..."
        
        # Install transformers and torch
        echo "📦 Installing PyTorch and transformers..."
        pip install torch transformers accelerate
        
        echo "✅ Hugging Face setup complete!"
        echo "⚠️  Note: First run will download models (~1-2GB)"
        
        export LLM_TYPE="huggingface"
        ;;
    *)
        echo "Invalid choice, defaulting to rule-based"
        export LLM_TYPE="rules"
        ;;
esac

echo ""
echo "🔧 Environment configured:"
echo "LLM_TYPE: $LLM_TYPE"
if [ "$LLM_TYPE" = "ollama" ]; then
    echo "OLLAMA_HOST: $OLLAMA_HOST"
fi

echo ""
echo "🚀 Ready to deploy! Run:"
echo "   export LLM_TYPE='$LLM_TYPE'"
if [ "$LLM_TYPE" = "ollama" ]; then
    echo "   export OLLAMA_HOST='$OLLAMA_HOST'"
fi
echo "   ./setup.sh"

# Test the selected setup
echo ""
echo "🧪 Testing LLM setup..."

if [ "$LLM_TYPE" = "ollama" ]; then
    if curl -s http://localhost:11434/api/tags | grep -q "models"; then
        echo "✅ Ollama is working!"
    else
        echo "❌ Ollama test failed. Make sure 'ollama serve' is running"
    fi
elif [ "$LLM_TYPE" = "huggingface" ]; then
    python3 -c "
try:
    from transformers import pipeline
    print('✅ Hugging Face transformers working!')
except ImportError:
    print('❌ Please install: pip install torch transformers accelerate')
" 2>/dev/null
fi

echo "🎯 Setup complete! Your incident response system will use $LLM_TYPE for analysis."
