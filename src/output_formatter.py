# src/output_formatter.py
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
# --- REMOVE RICH IMPORTS ---
# from rich import console, Panel

# Get the directory containing this script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume templates/ directory is at the same level as src/
_TEMPLATE_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'templates')

def generate_html_report(
        question,
        answer_text,
        evidence_spans_with_scores, # Expect list of (score, sentence) tuples
        full_doc_text,
        nli_judgment="N/A",
        nli_reasoning="(Reasoning not provided)", # Add argument with default
        confidence_score="N/A",
        output_filename="verification_report.html"
    ):
    """Generates a static HTML report visualizing the results, including NLI check."""
    print(f"Generating HTML report to {output_filename}...")

    # Set up Jinja2 environment
    try:
        env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            autoescape=select_autoescape(['html', 'xml']) # Enable autoescaping
        )
        template = env.get_template("report_template.html") # Ensure this template exists
    except Exception as e:
        print(f"Error loading Jinja2 template: {e}")
        print(f"Ensure '{_TEMPLATE_DIR}/report_template.html' exists.")
        return

    # Prepare data for the template
    # Check if evidence is placeholder/error message
    is_placeholder = False
    evidence_for_template = [] # Prepare list for template

    if isinstance(evidence_spans_with_scores, list) and evidence_spans_with_scores:
        # Check if the first item's text (index 1) matches typical placeholders
        first_span = evidence_spans_with_scores[0]
        if isinstance(first_span, (list, tuple)) and len(first_span) > 1:
            first_span_text = first_span[1]
            if isinstance(first_span_text, str) and \
               any(msg in first_span_text for msg in ["(Attention", "(Cannot process", "(Could not prepare", "(No evidence spans", "(Model inference failed", "(Pipeline failed", "(Processing error"]):
                is_placeholder = True
                evidence_for_template = [(0.0, first_span_text)] # Pass placeholder text
            else:
                 # It's likely real evidence
                 evidence_for_template = evidence_spans_with_scores
        else:
             # Handle case where evidence might be just a list of strings (e.g., error messages)
             is_placeholder = True
             evidence_for_template = [(0.0, str(first_span))]

    elif not evidence_spans_with_scores:
         is_placeholder = True
         evidence_for_template = [(0.0, "No evidence spans identified.")]
    else: # Should not happen if input is always list, but handle anyway
         is_placeholder = True
         evidence_for_template = [(0.0, "Error: Unexpected evidence format.")]


    # --- ADD NLI results to template data ---
    template_data = {
        "question": question,
        "answer_text": answer_text,
        "full_doc_text": full_doc_text,
        "evidence_spans_with_scores": evidence_for_template, # Use the prepared list
        "is_placeholder": is_placeholder,
        "nli_judgment": nli_judgment,
        "nli_reasoning": nli_reasoning,
        "confidence_score": f"{confidence_score:.2f}" if isinstance(confidence_score, float) else confidence_score, # Format score
    }

    # Render the template
    try:
        html_content = template.render(template_data)
    except Exception as e:
        print(f"Error rendering Jinja2 template: {e}")
        return

    # Save the rendered HTML to a file
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Successfully generated {output_filename}")
    except Exception as e:
        print(f"Error writing HTML file {output_filename}: {e}")


def display_results(question, answer_text, evidence_spans_with_scores, nli_judgment="N/A", confidence_score="N/A"):
    """Formats and prints the final results to the console."""

    print("\n" + "="*25 + " Verification Results " + "="*25)
    print(f"Question: {question}")
    print(f"\nModel Answer:\n{answer_text}")

    print("\nPotential Evidence Spans (Based on Attention Analysis):")
    if isinstance(evidence_spans_with_scores, list) and evidence_spans_with_scores:
        # Check for placeholder messages specifically
        is_placeholder = any("(Attention" in span or "(Cannot process" in span or "(Could not prepare" in span for span in evidence_spans_with_scores if isinstance(span, str))
        if is_placeholder:
             print(evidence_spans_with_scores[0]) # Print the single placeholder/error message
        else:
            # Print formatted evidence
            for i, span in enumerate(evidence_spans_with_scores):
                print(f"Evidence {i+1}:\n---\n{span}\n---")
    elif not evidence_spans_with_scores: # Handles empty list specifically
         print("No evidence spans identified.")
    else: # Handle non-list or unexpected content
         print(f"Could not identify evidence spans (unexpected format: {type(evidence_spans_with_scores)})")

    print("="* (50 + len(" Verification Results ")) )
