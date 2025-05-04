# src/output_formatter.py
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import json # Needed for safe JS embedding if not using |tojson filter

# Get the directory containing this script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'templates') # Assumes templates/ is sibling to src/

def generate_html_report(
        question,
        answer_text,
        evidence_spans_with_scores, # Expect list of (score, sentence) tuples
        full_doc_text,
        output_filename="verification_report.html"
    ):
    """Generates a static HTML report visualizing the results."""
    print(f"Generating HTML report to {output_filename}...")

    # Set up Jinja2 environment
    try:
        env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            autoescape=select_autoescape(['html', 'xml']) # Enable autoescaping
        )
        template = env.get_template("report_template.html")
    except Exception as e:
        print(f"Error loading Jinja2 template: {e}")
        print(f"Ensure '{_TEMPLATE_DIR}/report_template.html' exists.")
        return

    # Prepare data for the template
    # Check if evidence is placeholder/error message
    is_placeholder = False
    if isinstance(evidence_spans_with_scores, list) and evidence_spans_with_scores:
        # Check if the first item's text matches typical placeholders
        first_span_text = evidence_spans_with_scores[0][1] if len(evidence_spans_with_scores[0]) > 1 else evidence_spans_with_scores[0]
        if isinstance(first_span_text, str) and \
           any(msg in first_span_text for msg in ["(Attention", "(Cannot process", "(Could not prepare", "(No evidence spans", "(Model inference failed"]):
            is_placeholder = True
            # Ensure it's a list containing one tuple for the template
            evidence_spans_with_scores = [(0.0, first_span_text)]
    elif not evidence_spans_with_scores:
         is_placeholder = True
         evidence_spans_with_scores = [(0.0, "No evidence spans identified.")]


    template_data = {
        "question": question,
        "answer_text": answer_text,
        "full_doc_text": full_doc_text,
        # Pass the list of tuples directly; Jinja's |tojson filter handles safe embedding
        "evidence_spans_with_scores": evidence_spans_with_scores if not is_placeholder else [],
        "is_placeholder": is_placeholder, # Pass boolean flag for conditional rendering
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
