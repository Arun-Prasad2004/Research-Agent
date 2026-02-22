"""
Example script demonstrating programmatic usage of SR-MARE.
"""

from sr_mare.core.orchestrator import ResearchOrchestrator


def main():
    """Run example research queries."""
    
    # Initialize orchestrator
    print("🚀 Initializing SR-MARE...")
    orchestrator = ResearchOrchestrator(
        max_iterations=3,
        confidence_threshold=0.75
    )
    
    # Test connections
    print("\n🔌 Testing connections...")
    if not orchestrator.test_connections():
        print("❌ Connection test failed. Make sure Ollama is running.")
        print("Run: ollama serve")
        return
    
    # Load sample documents
    print("\n📚 Loading sample documents...")
    with open("sr_mare/data/documents.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
    orchestrator.load_documents(documents)
    
    # Example questions
    questions = [
        "What are the main causes of climate change?",
        "How does machine learning work?",
        "What is quantum entanglement?"
    ]
    
    print(f"\n🔬 Running {len(questions)} research queries...\n")
    
    # Run research for each question
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}/{len(questions)}: {question}")
        print('='*70)
        
        try:
            result = orchestrator.research(question, top_k=5)
            results.append(result)
            
            # Print summary
            print(f"\n📊 SUMMARY:")
            print(f"  Confidence: {result['confidence_score']:.3f}")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Duration: {result['duration_seconds']:.2f}s")
            print(f"\n💡 Answer Preview:")
            answer_preview = result['final_answer'][:200] + "..."
            print(f"  {answer_preview}")
            
            # Save individual result
            output_file = f"example_result_{i}.txt"
            orchestrator.save_result(result, output_file)
            print(f"\n💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            continue
    
    # Print aggregate statistics
    print(f"\n\n{'='*70}")
    print("📈 AGGREGATE STATISTICS")
    print('='*70)
    
    stats = orchestrator.get_summary_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Example complete!")
    print(f"   Processed {len(results)} questions successfully")
    print(f"   Check example_result_*.txt for detailed results")


if __name__ == "__main__":
    main()
