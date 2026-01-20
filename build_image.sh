#!/bin/bash
set -e

echo "🐳 Building Docker image for LLM Incident Response System"

# Get LLM type
LLM_TYPE=${LLM_TYPE:-"rules"}
echo "🧠 Building for LLM type: $LLM_TYPE"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t llm-incident-response:latest .

# For minikube, we need to use minikube's Docker daemon
if command -v minikube &> /dev/null; then
    echo "🚀 Using minikube Docker daemon..."
    eval $(minikube docker-env)
    
    # Build again in minikube's context
    docker build -t llm-incident-response:latest .
    
    echo "✅ Image built in minikube registry"
    echo "📋 Available images in minikube:"
    docker images | grep llm-incident-response
else
    echo "⚠️  Minikube not found, building locally"
    echo "📝 You may need to push to a registry or import to minikube"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Deploy with: ./setup.sh"
echo "2. The deployment will use the image: llm-incident-response:latest"
echo "3. No ConfigMap mounting needed - code is baked into image!"
