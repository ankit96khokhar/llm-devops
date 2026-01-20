#!/usr/bin/env python3
"""
Demo script to showcase the FREE intelligent rule-based analysis
Run this to see how the system analyzes different incident patterns
"""

import asyncio
from datetime import datetime
from incident_analyzer import (
    IncidentContext, IncidentSeverity, LLMIncidentAnalyzer, RemediationAction
)

def create_test_incident(name: str, resource_type: str, description: str, logs: list, metrics: dict) -> IncidentContext:
    """Create a test incident for analysis"""
    return IncidentContext(
        timestamp=datetime.now(),
        namespace="default",
        resource_type=resource_type,
        resource_name=name,
        severity=IncidentSeverity.HIGH,
        description=description,
        logs=logs,
        metrics=metrics,
        events=[]
    )

async def demo_analysis():
    """Demonstrate the FREE intelligent analysis"""
    print("🆓 FREE Version - Intelligent Rule-Based Analysis Demo")
    print("=" * 55)
    
    # Initialize the analyzer (no API key needed)
    analyzer = LLMIncidentAnalyzer()
    
    # Test Case 1: CrashLoopBackOff
    print("\n🔍 Test Case 1: CrashLoopBackOff Detection")
    crash_incident = create_test_incident(
        name="crashy-app-12345",
        resource_type="Pod", 
        description="Pod is in CrashLoopBackOff state with 8 restarts",
        logs=[
            "2024-01-20 10:30:00 Starting application...",
            "2024-01-20 10:30:01 Connecting to database...", 
            "2024-01-20 10:30:02 ERROR: Connection refused",
            "2024-01-20 10:30:03 panic: cannot connect to database",
            "2024-01-20 10:30:03 exit code 1"
        ],
        metrics={"restart_count": 8, "ready_replicas": 0, "desired_replicas": 1}
    )
    
    plan = await analyzer.analyze_incident(crash_incident)
    print(f"   🎯 Action: {plan.action.value}")
    print(f"   📊 Confidence: {plan.confidence_score:.2f}")
    print(f"   💭 Reasoning: {plan.reasoning}")
    print(f"   📈 Impact: {plan.estimated_impact}")
    
    # Test Case 2: OOMKilled
    print("\n🔍 Test Case 2: Out of Memory Detection")
    oom_incident = create_test_incident(
        name="memory-hungry-app",
        resource_type="Pod",
        description="Container was OOMKilled due to memory limits", 
        logs=[
            "2024-01-20 10:25:00 Application starting...",
            "2024-01-20 10:25:30 Loading large dataset...",
            "2024-01-20 10:26:00 WARNING: Memory usage at 95%",
            "2024-01-20 10:26:15 FATAL: out of memory",
            "2024-01-20 10:26:15 Process killed"
        ],
        metrics={"restart_count": 3, "ready_replicas": 0, "desired_replicas": 2}
    )
    
    plan = await analyzer.analyze_incident(oom_incident)
    print(f"   🎯 Action: {plan.action.value}")
    print(f"   📊 Confidence: {plan.confidence_score:.2f}")
    print(f"   💭 Reasoning: {plan.reasoning}")
    print(f"   📈 Impact: {plan.estimated_impact}")
    
    # Test Case 3: ImagePullBackOff
    print("\n🔍 Test Case 3: Image Pull Failure Detection")
    image_incident = create_test_incident(
        name="bad-image-pod",
        resource_type="Pod",
        description="Pod is in ImagePullBackOff state",
        logs=[
            "2024-01-20 10:20:00 Pulling image registry.example.com/nonexistent:latest",
            "2024-01-20 10:20:05 ERROR: pull access denied for nonexistent",
            "2024-01-20 10:20:05 repository does not exist or may require docker login"
        ],
        metrics={"restart_count": 0, "ready_replicas": 0, "desired_replicas": 1}
    )
    
    plan = await analyzer.analyze_incident(image_incident)
    print(f"   🎯 Action: {plan.action.value}")
    print(f"   📊 Confidence: {plan.confidence_score:.2f}")
    print(f"   💭 Reasoning: {plan.reasoning}")
    print(f"   📈 Impact: {plan.estimated_impact}")
    
    # Test Case 4: Deployment Failure
    print("\n🔍 Test Case 4: Deployment Scaling Issues")
    deployment_incident = create_test_incident(
        name="failing-deployment",
        resource_type="Deployment",
        description="Deployment has 0 ready replicas out of 3 desired",
        logs=[
            "2024-01-20 10:15:00 Deployment started",
            "2024-01-20 10:15:30 Scaling up to 3 replicas", 
            "2024-01-20 10:16:00 All pods failing to start"
        ],
        metrics={"restart_count": 2, "ready_replicas": 0, "desired_replicas": 3}
    )
    
    plan = await analyzer.analyze_incident(deployment_incident)
    print(f"   🎯 Action: {plan.action.value}")
    print(f"   📊 Confidence: {plan.confidence_score:.2f}")
    print(f"   💭 Reasoning: {plan.reasoning}")
    print(f"   📈 Impact: {plan.estimated_impact}")
    
    print("\n" + "=" * 55)
    print("✨ Demo Complete! This shows how the FREE version provides intelligent")
    print("   analysis without any external APIs or costs.")
    print("🧠 The system uses pattern matching, keyword analysis, and metric")
    print("   evaluation to make smart remediation decisions.")
    print("🆓 Zero cost, maximum value for your learning environment!")

if __name__ == "__main__":
    asyncio.run(demo_analysis())
