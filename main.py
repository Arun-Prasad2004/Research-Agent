"""
SR-MARE: Self-Reflective Multi-Agent Research Engine
Main CLI interface for running research queries.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from sr_mare.core.orchestrator import ResearchOrchestrator


def load_documents_from_file(file_path: str) -> List[str]:
    """
    Load documents from a text file.
    Each document should be separated by double newlines.
    
    Args:
        file_path: Path to the documents file
        
    Returns:
        List of document strings
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"⚠ Warning: Document file not found at {file_path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines or by specific delimiter
    documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
    
    if not documents:
        # Try splitting by single newlines if no double newlines found
        documents = [doc.strip() for doc in content.split('\n') if doc.strip()]
    
    print(f"📚 Loaded {len(documents)} documents from {file_path}")
    return documents


def print_formatted_report(result: dict):
    """
    Print a formatted research report to console.
    
    Args:
        result: Research result dictionary
    """
    print("\n" + "=" * 80)
    print("🔬 SR-MARE RESEARCH REPORT")
    print("=" * 80)
    
    print(f"\n📝 QUESTION:")
    print(f"  {result['question']}")
    
    print(f"\n🎯 TASK BREAKDOWN:")
    plan = result['task_breakdown']
    if isinstance(plan, dict):
        if 'subtasks' in plan:
            for i, task in enumerate(plan['subtasks'], 1):
                print(f"  {i}. {task}")
        if 'key_concepts' in plan:
            print(f"\n  Key Concepts: {', '.join(plan['key_concepts'])}")
    
    print(f"\n💡 FINAL ANSWER:")
    print(f"  {result['final_answer']}")
    
    print(f"\n📊 CONFIDENCE METRICS:")
    conf = result['confidence_metrics']
    print(f"  Overall Confidence:       {conf['final_confidence']:.3f}")
    print(f"  Critic Quality Score:     {conf['critic_quality_score']:.3f}")
    print(f"  Self-Consistency:         {conf['self_consistency_score']:.3f}")
    print(f"  Evidence Diversity:       {conf['evidence_diversity_score']:.3f}")
    print(f"  Retrieval Quality:        {conf['retrieval_quality']:.3f}")
    
    print(f"\n🎯 CRITIC FEEDBACK:")
    critique = result['critic_feedback']
    if 'strengths' in critique and critique['strengths']:
        print(f"  Strengths:")
        for strength in critique['strengths']:
            print(f"    ✓ {strength}")
    if 'weaknesses' in critique and critique['weaknesses']:
        print(f"  Areas for Improvement:")
        for weakness in critique['weaknesses']:
            print(f"    • {weakness}")
    print(f"  Hallucination Risk:       {critique.get('hallucination_risk', 'unknown')}")
    
    print(f"\n🔄 ITERATION METRICS:")
    print(f"  Total Iterations:         {result['iterations']}")
    iter_metrics = result['iteration_metrics']
    print(f"  Total Improvement:        {iter_metrics.get('total_improvement', 0):.3f}")
    print(f"  Converged:                {iter_metrics.get('converged', False)}")
    
    print(f"\n📚 RETRIEVAL METRICS:")
    retr = result['retrieval_metrics']
    print(f"  Documents Retrieved:      {retr['num_retrieved']}")
    print(f"  Average Similarity:       {retr['avg_similarity']:.3f}")
    print(f"  Relevant Documents:       {retr['num_relevant']}")
    
    print(f"\n🔍 RETRIEVED SOURCES:")
    for i, source in enumerate(result['retrieved_sources'][:3], 1):
        print(f"  [{i}] Similarity: {source['similarity']:.3f}")
        print(f"      {source['text'][:150]}...")
    
    print(f"\n⏱ PROCESSING TIME: {result['duration_seconds']:.2f} seconds")
    
    print("\n" + "=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SR-MARE: Self-Reflective Multi-Agent Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a research query
  python main.py --question "What are the key factors in climate change?"
  
  # Load custom documents
  python main.py --question "Your question" --documents path/to/docs.txt
  
  # Adjust confidence threshold
  python main.py --question "Your question" --threshold 0.8
  
  # Save to custom output file
  python main.py --question "Your question" --output my_results.txt
        """
    )
    
    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Research question to answer'
    )
    
    parser.add_argument(
        '--documents', '-d',
        type=str,
        default='sr_mare/data/documents.txt',
        help='Path to documents file (default: sr_mare/data/documents.txt)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results.txt',
        help='Output file path for results (default: results.txt)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.75,
        help='Confidence threshold for stopping refinement (default: 0.75)'
    )
    
    parser.add_argument(
        '--max-iterations', '-m',
        type=int,
        default=3,
        help='Maximum refinement iterations (default: 3)'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test Ollama connection and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    print("🚀 Initializing SR-MARE...")
    orchestrator = ResearchOrchestrator(
        base_url=args.ollama_url,
        max_iterations=args.max_iterations,
        confidence_threshold=args.threshold
    )
    
    # Test connection if requested
    if args.test_connection:
        print("\n🔌 Testing connections...")
        success = orchestrator.test_connections()
        sys.exit(0 if success else 1)
    
    # Load documents
    documents = load_documents_from_file(args.documents)
    
    if documents:
        orchestrator.load_documents(documents)
    else:
        print("⚠ Warning: No documents loaded. Results may be limited.")
    
    # Get question from args or prompt user
    question = args.question
    
    if not question:
        print("\n" + "=" * 80)
        print("Enter your research question (or 'quit' to exit):")
        question = input("> ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            sys.exit(0)
    
    if not question:
        print("Error: No question provided")
        sys.exit(1)
    
    # Run research
    try:
        result = orchestrator.research(question, top_k=args.top_k)
        
        # Print formatted report
        print_formatted_report(result)
        
        # Save to file
        orchestrator.save_result(result, args.output)
        print(f"\n💾 Full results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
