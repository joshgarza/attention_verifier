# src/output_formatter.py

def display_results(question, answer_text, evidence_spans):
    """Formats and prints the final results to the console."""

    print("\n" + "="*25 + " Verification Results " + "="*25)
    print(f"Question: {question}")
    print(f"\nModel Answer:\n{answer_text}")

    print("\nPotential Evidence Spans (Based on Attention Analysis):")
    if isinstance(evidence_spans, list) and evidence_spans:
        # Check for placeholder messages specifically
        is_placeholder = any("(Attention" in span or "(Cannot process" in span or "(Could not prepare" in span for span in evidence_spans if isinstance(span, str))
        if is_placeholder:
             print(evidence_spans[0]) # Print the single placeholder/error message
        else:
            # Print formatted evidence
            for i, span in enumerate(evidence_spans):
                print(f"Evidence {i+1}:\n---\n{span}\n---")
    elif not evidence_spans: # Handles empty list specifically
         print("No evidence spans identified.")
    else: # Handle non-list or unexpected content
         print(f"Could not identify evidence spans (unexpected format: {type(evidence_spans)})")

    print("="* (50 + len(" Verification Results ")) )