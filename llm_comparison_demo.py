#!/usr/bin/env python3
"""
Advanced Local LLM Demo - Compare different analysis methods
Shows rule-based vs Ollama vs Hugging Face analysis
"""

import asyncio
import os
from datetime import datetime
from incident_analyzer import (
    IncidentContext, IncidentSeverity, LLMIncidentAnalyzer, RemediationAction
)

def create_test_incident() -> IncidentContext:
    """Create a complex test incident"""
    return IncidentContext(
        timestamp=datetime.now(),
        namespace="production",
        resource_type="Pod",
        resource_name="payment-service-7d8f9k2",
        severity=IncidentSeverity.HIGH,
        description="Pod in CrashLoopBackOff with frequent restarts and OOM kills",
        logs=[
            "2024-01-20 14:30:00 INFO Starting payment service v2.1.4",
            "2024-01-20 14:30:05 INFO Connecting to PostgreSQL database...",
            "2024-01-20 14:30:10 INFO Loading customer cache (500MB)...", 
            "2024-01-20 14:30:45 WARN Memory usage at 90% (450MB/500MB)",
            "2024-01-20 14:31:00 ERROR OutOfMemoryError: Java heap space",
            "2024-01-20 14:31:00 FATAL Application crashed with exit code 137",
            "2024-01-20 14:31:15 INFO Restarting container attempt #8...",
            "2024-01-20 14:31:20 INFO JVM startup with -Xmx512m",
            "2024-01-20 14:31:25 ERROR Cannot allocate memory for heap"
        ],
        metrics={
            "restart_count": 8,
            "ready_replicas": 0,
            "desired_replicas": 3,
            "cpu_usage": "250m",
            "memory_usage": "500Mi/500Mi"
        },
        events=[
            {
                "reason": "BackOff",
                "message": "Back-off restarting failed container",
                "type": "Warning",
                "timestamp": datetime.now(),
                "count": 8
            }
        ]
    )

async def compare_analysis_methods():
    """Compare all three analysis methods"""
    print("🧪 LOCAL LLM COMPARISON DEMO")
    print("=" * 60)
    
    incident = create_test_incident()
    
    print(f"\n📋 ANALYZING INCIDENT:")
    print(f"Resource: {incident.resource_name}")
    print(f"Issue: {incident.description}")
    print(f"Restart Count: {incident.metrics['restart_count']}")
    print(f"Memory: {incident.metrics['memory_usage']}")
    print()
    
    # Test 1: Rule-based analysis
    print("🔍 METHOD 1: Rule-Based Analysis (FREE)")
    print("-" * 40)
    
    rules_analyzer = LLMIncidentAnalyzer(llm_type="rules")
    rules_plan = await rules_analyzer.analyze_incident(incident)
    
    print(f"   🎯 Action: {rules_plan.action.value}")
    print(f"   📊 Confidence: {rules_plan.confidence_score:.2f}")
    print(f"   💭 Reasoning: {rules_plan.reasoning}")
    print(f"   ⚡ Speed: Instant")
    print(f"   💰 Cost: $0")
    
    # Test 2: Ollama analysis (if available)
    print("\n🔍 METHOD 2: Ollama + Llama Analysis")
    print("-" * 40)
    
    try:
        ollama_analyzer = LLMIncidentAnalyzer(llm_type="ollama")
        if ollama_analyzer.llm_type == "ollama":
            ollama_plan = await ollama_analyzer.analyze_incident(incident)
            print(f"   🎯 Action: {ollama_plan.action.value}")
            print(f"   📊 Confidence: {ollama_plan.confidence_score:.2f}")
            print(f"   💭 Reasoning: {ollama_plan.reasoning}")
            print(f"   ⚡ Speed: ~2-5 seconds")
            print(f"   💰 Cost: $0 (local)")
        else:
            print("   ❌ Ollama not available (install: curl -fsSL https://ollama.ai/install.sh | sh)")
            print("   📝 Then run: ollama serve && ollama pull llama3.2")
    except Exception as e:
        print(f"   ❌ Ollama error: {e}")
    
    # Test 3: Hugging Face analysis (if available) 
    print("\n🔍 METHOD 3: Hugging Face Transformers")
    print("-" * 40)
    
    try:
        hf_analyzer = LLMIncidentAnalyzer(llm_type="huggingface")
        if hf_analyzer.llm_type == "huggingface":
            hf_plan = await hf_analyzer.analyze_incident(incident)
            print(f"   🎯 Action: {hf_plan.action.value}")
            print(f"   📊 Confidence: {hf_plan.confidence_score:.2f}")
            print(f"   💭 Reasoning: {hf_plan.reasoning}")
            print(f"   ⚡ Speed: ~10-30 seconds")
            print(f"   💰 Cost: $0 (local)")
        else:
            print("   ❌ Transformers not available (install: pip install torch transformers)")
    except Exception as e:
        print(f"   ❌ Hugging Face error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison_table = """
╔═══════════════════╦══════════════╦═══════════╦══════════════╦═════════════╗
║ Method            ║ Speed        ║ Cost      ║ Accuracy     ║ Setup       ║
╠═══════════════════╬══════════════╬═══════════╬══════════════╬═════════════╣
║ Rules             ║ Instant      ║ $0        ║ 75-85%       ║ None        ║
║ Ollama (Llama)    ║ 2-5 sec      ║ $0        ║ 85-95%       ║ Easy        ║
║ Hugging Face      ║ 10-30 sec    ║ $0        ║ 80-90%       ║ Medium      ║
║ OpenAI GPT-4      ║ 1-3 sec      ║ ~$0.003   ║ 90-98%       ║ API key     ║
╚═══════════════════╩══════════════╩═══════════╩══════════════╩═════════════╝
    """
    print(comparison_table)
    
    print("\n🎯 RECOMMENDATIONS:")
    print("   🆓 Learning/Dev: Use Rules (fast, free, educational)")
    print("   🦙 Production: Use Ollama + Llama3.2 (best balance)")
    print("   🤗 Experimentation: Use Hugging Face (many model options)")
    print("   💳 Enterprise: Consider OpenAI (highest accuracy)")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Choose your preferred method")
    print("   2. Run: export LLM_TYPE='rules|ollama|huggingface'")  
    print("   3. Deploy: ./setup.sh")
    print("   4. Test: ./create_test_scenarios.sh")

async def interactive_setup():
    """Interactive LLM setup"""
    print("\n🔧 INTERACTIVE SETUP")
    print("Which LLM method do you want to use?")
    print("1. 🆓 Rules (instant, free)")
    print("2. 🦙 Ollama (best for learning)")
    print("3. 🤗 Hugging Face (most options)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "2":
        print("\n🦙 Setting up Ollama...")
        print("Run these commands:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("   ollama serve &")
        print("   ollama pull llama3.2")
        print("   export LLM_TYPE='ollama'")
        
    elif choice == "3":
        print("\n🤗 Setting up Hugging Face...")
        print("Run these commands:")
        print("   pip install torch transformers accelerate")
        print("   export LLM_TYPE='huggingface'")
        
    else:
        print("\n🆓 Using rule-based analysis (default)")
        print("   export LLM_TYPE='rules'")
        
    print("\nThen deploy with: ./setup.sh")

if __name__ == "__main__":
    print("🤖 Welcome to Advanced Local LLM Demo!")
    
    mode = input("\nChoose mode:\n1. 🧪 Compare methods\n2. 🔧 Interactive setup\n\nEnter choice (1-2): ").strip()
    
    if mode == "2":
        asyncio.run(interactive_setup())
    else:
        asyncio.run(compare_analysis_methods())
