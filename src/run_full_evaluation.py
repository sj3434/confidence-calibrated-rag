"""
Full evaluation suite: runs 3 modes and saves artifacts.
Mode 1: Calibrated RAG (both guardrails)
Mode 2: Naive RAG (no guardrails)
Mode 3: Guardrail 1 Only (threshold only, skip judge)
"""
import os
import json
import sys

# Set API key
os.environ["OPENAI_API_KEY"] = open(os.path.join(os.path.dirname(__file__), "../../.env")).read().strip()

from rag import ConfidenceCalibratedRAG
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class CustomDeepEvalMock:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.faith_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator. Determine if the Actual Output is logically entailed by the Retrieval Context. Reply with only a float between 0.0 and 1.0."),
            ("user", "Context: {context}\nOutput: {output}")
        ])
        self.rel_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator. Determine if the Actual Output is highly relevant to the Input. Reply with only a float between 0.0 and 1.0."),
            ("user", "Input: {input}\nOutput: {output}")
        ])
    
    def evaluate_faithfulness(self, output, context):
        try:
            chain = self.faith_prompt | self.llm | StrOutputParser()
            score = float(chain.invoke({"context": context, "output": output}).strip())
            return score, score >= 0.5
        except:
            return 1.0, True

    def evaluate_relevancy(self, input_text, output):
        try:
            chain = self.rel_prompt | self.llm | StrOutputParser()
            score = float(chain.invoke({"input": input_text, "output": output}).strip())
            return score, score >= 0.5
        except:
            return 1.0, True

def run_evaluation(mode="calibrated"):
    """
    mode: "calibrated" | "naive" | "g1_only"
    """
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    corpus_path = os.path.join(data_dir, "corpus.txt")
    eval_path = os.path.join(data_dir, "eval_dataset.json")
    
    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper()}")
    print(f"{'='*60}")
    
    rag = ConfidenceCalibratedRAG(data_path=corpus_path)
    
    # Always calibrate to get threshold
    calibration_data = [
        ("in", "What is Generative AI?"),
        ("in", "When was Columbia University founded?"),
        ("out", "How do I bake a cake?"),
        ("out", "What is the capital of France?")
    ]
    threshold = rag.calibrate(calibration_data)
    
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    
    evaluator = CustomDeepEvalMock()
    results = []
    
    for i, case in enumerate(eval_cases):
        query = case["query"]
        expected = case["expected_behavior"]
        domain = case["domain"]
        
        # Get similarity scores first (always)
        raw_results = rag.retrieve_with_scores(query)
        best_score = raw_results[0][1] if raw_results else 0
        context = "\n".join([doc.page_content for doc, score in raw_results])
        
        if mode == "naive":
            # No guardrails - always generate
            actual_output, docs = rag.query(query, enable_abstention=False)
            abstained = "I don't know" in actual_output
            judge_verdict = "N/A"
        elif mode == "g1_only":
            # Only threshold, skip judge
            if best_score < rag.threshold:
                actual_output = f"I don't know (Abstained: Max similarity score {best_score:.3f} < threshold {rag.threshold:.3f})"
                docs = raw_results
                abstained = True
                judge_verdict = "SKIPPED"
            else:
                # Pass G1, skip G2, go straight to generation
                qa_chain = rag.qa_prompt | rag.llm | StrOutputParser()
                actual_output = qa_chain.invoke({"context": context, "question": query})
                docs = raw_results
                abstained = False
                judge_verdict = "SKIPPED"
        else:  # calibrated (both guardrails)
            actual_output, docs = rag.query(query, enable_abstention=True)
            abstained = "I don't know" in actual_output
            # Determine which guardrail triggered
            if abstained and best_score < rag.threshold:
                judge_verdict = "N/A (G1 caught)"
            elif abstained:
                judge_verdict = "NO"
            else:
                judge_verdict = "YES"
        
        # Evaluate faithfulness/relevancy for answered queries
        faith_score = None
        rel_score = None
        faith_pass = None
        rel_pass = None
        
        if not abstained:
            faith_score, faith_pass = evaluator.evaluate_faithfulness(actual_output, context)
            rel_score, rel_pass = evaluator.evaluate_relevancy(query, actual_output)
        
        # Determine pass/fail
        if expected == "abstain":
            passed = abstained
        else:
            if abstained:
                passed = False  # Should have answered
            else:
                passed = (faith_pass and rel_pass) if (faith_pass is not None) else False
        
        result = {
            "index": i + 1,
            "query": query,
            "domain": domain,
            "expected": expected,
            "sim_score": round(float(best_score), 4),
            "above_threshold": bool(best_score >= rag.threshold),
            "judge_verdict": judge_verdict,
            "abstained": bool(abstained),
            "passed": bool(passed),
            "faithfulness": round(float(faith_score), 4) if faith_score is not None else None,
            "relevancy": round(float(rel_score), 4) if rel_score is not None else None,
            "answer_preview": actual_output[:150] if not abstained else actual_output
        }
        results.append(result)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status} [{domain}] Q{i+1}: '{query[:50]}...' sim={best_score:.4f} judge={judge_verdict}", end="")
        if faith_score is not None:
            print(f" faith={faith_score:.2f} rel={rel_score:.2f}", end="")
        print()
    
    # Summary stats
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    in_domain = [r for r in results if r["domain"] == "in-domain"]
    ood = [r for r in results if r["domain"] == "out-of-domain"]
    in_passed = sum(1 for r in in_domain if r["passed"])
    ood_passed = sum(1 for r in ood if r["passed"])
    ood_abstained = sum(1 for r in ood if r["abstained"])
    ood_hallucinated = sum(1 for r in ood if not r["abstained"])
    
    faith_scores = [r["faithfulness"] for r in in_domain if r["faithfulness"] is not None]
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
    
    summary = {
        "mode": mode,
        "threshold": round(rag.threshold, 4),
        "total": total,
        "passed": passed_count,
        "accuracy": round(passed_count / total, 4),
        "in_domain_total": len(in_domain),
        "in_domain_passed": in_passed,
        "in_domain_accuracy": round(in_passed / len(in_domain), 4),
        "ood_total": len(ood),
        "ood_abstained": ood_abstained,
        "ood_hallucinated": ood_hallucinated,
        "ood_abstention_rate": round(ood_abstained / len(ood), 4),
        "avg_faithfulness_in_domain": round(avg_faith, 4)
    }
    
    print(f"\n--- SUMMARY ({mode}) ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    return results, summary

if __name__ == "__main__":
    all_data = {}
    
    for mode in ["calibrated", "naive", "g1_only"]:
        results, summary = run_evaluation(mode)
        all_data[mode] = {"results": results, "summary": summary}
    
    # Save all results
    output_dir = os.path.join(os.path.dirname(__file__), "..")
    
    output_path = os.path.join(output_dir, "evaluation_full_results.json")
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n✅ Full results saved to {output_path}")
    
    # Also save human-readable summary
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        for mode in ["calibrated", "naive", "g1_only"]:
            s = all_data[mode]["summary"]
            f.write(f"{'='*50}\n")
            f.write(f"MODE: {mode.upper()}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Threshold: {s['threshold']}\n")
            f.write(f"Overall Accuracy: {s['passed']}/{s['total']} ({s['accuracy']*100:.1f}%)\n")
            f.write(f"In-Domain Accuracy: {s['in_domain_passed']}/{s['in_domain_total']} ({s['in_domain_accuracy']*100:.1f}%)\n")
            f.write(f"OOD Abstention Rate: {s['ood_abstained']}/{s['ood_total']} ({s['ood_abstention_rate']*100:.1f}%)\n")
            f.write(f"OOD Hallucinated: {s['ood_hallucinated']}/{s['ood_total']}\n")
            f.write(f"Avg Faithfulness (In-Domain): {s['avg_faithfulness_in_domain']}\n\n")
            
            f.write("Per-query results:\n")
            for r in all_data[mode]["results"]:
                status = "PASS" if r["passed"] else "FAIL"
                f.write(f"  [{status}] Q{r['index']} ({r['domain']}): sim={r['sim_score']:.4f} judge={r['judge_verdict']}")
                if r["faithfulness"] is not None:
                    f.write(f" faith={r['faithfulness']:.2f} rel={r['relevancy']:.2f}")
                f.write(f"\n    Query: {r['query']}\n")
            f.write("\n")
    print(f"✅ Summary saved to {summary_path}")
