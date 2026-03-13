import os
import json
import asyncio
from pathlib import Path
from typing import Any
from datetime import datetime
import litellm

# Set litellm to output nicely
litellm.suppress_debug_info = True


from extract_bench import ReportBuilder, ReportConfig

def strip_evaluation_config(schema: Any) -> Any:
    """Recursively remove 'evaluation_config' from JSON schema."""
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k == "evaluation_config":
                continue
            cleaned[k] = strip_evaluation_config(v)
        return cleaned
    elif isinstance(schema, list):
        return [strip_evaluation_config(item) for item in schema]
    else:
        return schema

async def process_and_evaluate(setup: str, model_name: str):
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    schema_path = Path("dataset/academic/research/research-schema.json")
    base_dir = Path("dataset/academic/research/pdf+gold")
    out_base_dir = Path("outputs/academic")
    
    # Load schema
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    clean_schema = strip_evaluation_config(schema)
    
    pdf_files = list(base_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {base_dir}.")
        return

    run_output_dir = out_base_dir / f"{setup}_{model_name.replace('/', '_')}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store overall stats
    overall_stats = []

    for pdf_path in pdf_files:
        print(f"\n--- Processing {pdf_path.name} ({setup}) ---")
        gold_path = pdf_path.with_name(pdf_path.stem + ".gold.json")
        
        if not gold_path.exists():
            print(f"Gold JSON not found for {pdf_path.name}, skipping.")
            continue
            
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        
        predicted_path = run_output_dir / gold_path.name
        
        if not predicted_path.exists():
            print(f"Querying LLM ({model_name})...")
            
            schema_str = json.dumps(clean_schema, indent=2)
            
            if setup == "setup1":
                # Original setup: PDF (multimodal) + original prompt
                import base64
                
                # Note: passing PDF to litellm might require specific formatting depending on the provider.
                # Here we assume a typical data URI for multimodal inputs.
                with open(pdf_path, "rb") as f:
                    pdf_data = base64.b64encode(f.read()).decode("utf-8")
                    
                prompt_text = (
                    f"Using the JSON template as a guideline , extract all\n"
                    f"the required information from {pdf_path.name} document.\n"
                    f"JSON Template: {schema_str}\n"
                    f"Please return ONLY valid JSON that conforms to this\n"
                    f"schema. Do not include any explanatory text before or after the JSON."
                )
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_data}"}}
                        ]
                    }
                ]
                
            elif setup == "setup2":
                # Multimodal + Improved Setup: PDF (visual) + improved expert prompt
                import base64
                
                with open(pdf_path, "rb") as f:
                    pdf_data = base64.b64encode(f.read()).decode("utf-8")
                
                prompt_text = (
                    f"You are an expert academic data extractor with advanced document analysis skills. "
                    f"Your task is to extract highly accurate, structured metadata from this academic PDF.\n\n"
                    f"Document Name: {pdf_path.name}\n\n"
                    f"Your required extraction schema:\n{schema_str}\n\n"
                    f"CRITICAL METADATA EXTRACTION INSTRUCTIONS:\n"
                    f"1. PUBLICATION ID (ids field): Look carefully at margins, headers, or visual metadata.\n"
                    f"   - For arXiv papers: Check the grey vertical text on the left side or top margins.\n"
                    f"   - For DOI papers: Look in footer or opening pages.\n"
                    f"   - If not found, set to null.\n"
                    f"2. NUMBER OF PAGES: Count total pages or look for 'page X of Y' markers.\n"
                    f"3. PUBLICATION DATE: Extract from visible date markers, headers, or submission info.\n"
                    f"   - Format as YYYY-MM-DD, YYYY-MM, or YYYY.\n"
                    f"4. PUBLICATION TYPE: Infer from document structure and keywords (Conference, Journal, Preprint, etc.).\n"
                    f"5. VENUE: Extract conference/journal names from headers, first page, or metadata.\n"
                    f"6. GENERAL INSTRUCTIONS:\n"
                    f"   - Analyze the visual layout to find metadata in margins, headers, footers.\n"
                    f"   - Be thorough with author lists, keywords, and citations.\n"
                    f"   - If information is not visibly present, do NOT invent it. Set to null or omit.\n"
                    f"   - Return ONLY a valid, well-formatted JSON object adhering to the schema.\n"
                    f"   - Do NOT include Markdown code blocks. Output raw JSON only."
                )
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_data}"}}
                        ]
                    }
                ]
                
            elif setup == "setup3":
                # New Setup: Markdown + original prompt
                md_path = pdf_path.with_suffix(".md")
                if not md_path.exists():
                    print(f"Markdown not found for {pdf_path.name}, skipped (Did you run script 01?).")
                    continue
                md_content = md_path.read_text(encoding="utf-8")
                
                prompt_text = (
                    f"Using the JSON template as a guideline , extract all\n"
                    f"the required information from {pdf_path.name} document.\n\n"
                    f"{md_content}\n\n"
                    f"JSON Template: {schema_str}\n"
                    f"Please return ONLY valid JSON that conforms to this\n"
                    f"schema. Do not include any explanatory text before or after the JSON."
                )
                
                messages = [{"role": "user", "content": prompt_text}]
                
            elif setup == "setup4":
                # New Improved Setup: Markdown + improved prompt
                md_path = pdf_path.with_suffix(".md")
                if not md_path.exists():
                    print(f"Markdown not found for {pdf_path.name}, skipped (Did you run script 01?).")
                    continue
                md_content = md_path.read_text(encoding="utf-8")
                
                prompt_text = (
                    f"You are an expert academic data extractor with advanced document analysis skills. "
                    f"Your task is to extract highly accurate, structured metadata from this academic PDF.\n\n"
                    f"Document Name: {pdf_path.name}\n\n"
                    f"Your required extraction schema:\n{schema_str}\n\n"
                    f"CRITICAL METADATA EXTRACTION INSTRUCTIONS:\n"
                    f"1. PUBLICATION ID (ids field): Look carefully at margins, headers, or visual metadata.\n"
                    f"   - For arXiv papers: Check the grey vertical text on the left side or top margins.\n"
                    f"   - For DOI papers: Look in footer or opening pages.\n"
                    f"   - If not found, set to null.\n"
                    f"2. NUMBER OF PAGES: Count total pages or look for 'page X of Y' markers.\n"
                    f"3. PUBLICATION DATE: Extract from visible date markers, headers, or submission info.\n"
                    f"   - Format as YYYY-MM-DD, YYYY-MM, or YYYY.\n"
                    f"4. PUBLICATION TYPE: Infer from document structure and keywords (Conference, Journal, Preprint, etc.).\n"
                    f"5. VENUE: Extract conference/journal names from headers, first page, or metadata.\n"
                    f"6. GENERAL INSTRUCTIONS:\n"
                    f"   - Analyze the visual layout to find metadata in margins, headers, footers.\n"
                    f"   - Be thorough with author lists, keywords, and citations.\n"
                    f"   - If information is not visibly present, do NOT invent it. Set to null or omit.\n"
                    f"   - Return ONLY a valid, well-formatted JSON object adhering to the schema.\n"
                    f"   - Do NOT include Markdown code blocks. Output raw JSON only."
                    f"Document Content:\n"
                    f"{md_content}\n\n"
                )
               
                
                messages = [{"role": "user", "content": prompt_text}]

            
            
            elif setup == "setup5":
                # PDF + Markdown with original prompt
                import base64
                
                md_path = pdf_path.with_suffix(".md")
                if not md_path.exists():
                    print(f"Markdown not found for {pdf_path.name}, skipped (Did you run script 01?).")
                    continue
                md_content = md_path.read_text(encoding="utf-8")
                
                with open(pdf_path, "rb") as f:
                    pdf_data = base64.b64encode(f.read()).decode("utf-8")
                
                prompt_text = (
                    f"Using the JSON template as a guideline, extract all\n"
                    f"the required information from {pdf_path.name} document.\n\n"
                    f"Here is the extracted text content:\n"
                    f"{md_content}\n\n"
                    f"JSON Template: {schema_str}\n"
                    f"Please return ONLY valid JSON that conforms to this\n"
                    f"schema. Do not include any explanatory text before or after the JSON."
                )
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_data}"}}
                        ]
                    }
                ]
            
            elif setup == "setup6":
                # PDF + Markdown with improved prompt
                import base64
                
                md_path = pdf_path.with_suffix(".md")
                if not md_path.exists():
                    print(f"Markdown not found for {pdf_path.name}, skipped (Did you run script 01?).")
                    continue
                md_content = md_path.read_text(encoding="utf-8")
                
                with open(pdf_path, "rb") as f:
                    pdf_data = base64.b64encode(f.read()).decode("utf-8")
                
                prompt_text = (
                    f"You are an expert academic data extractor with advanced document analysis skills. "
                    f"Your task is to extract highly accurate, structured metadata from this academic document.\n\n"
                    f"Document Name: {pdf_path.name}\n\n"
                    f"You have access to:\n"
                    f"1. The full PDF document (visual layout, margins, headers, footers)\n"
                    f"2. The extracted text content in Markdown format\n\n"
                    f"Extracted Text Content:\n"
                    f"{md_content}\n\n"
                    f"Your required extraction schema:\n{schema_str}\n\n"
                    f"CRITICAL METADATA EXTRACTION INSTRUCTIONS:\n"
                    f"1. PUBLICATION ID (ids field): Look for arXiv IDs or DOI in both visible margins and text.\n"
                    f"   - Check visual metadata (side margins), headers, and body text.\n"
                    f"   - If not found, set to null.\n"
                    f"2. NUMBER OF PAGES: Use visual page count from PDF or count from document structure.\n"
                    f"3. PUBLICATION DATE: Extract from visible date markers, headers, or metadata.\n"
                    f"   - Format as YYYY-MM-DD, YYYY-MM, or YYYY.\n"
                    f"4. PUBLICATION TYPE: Infer from document structure and keywords (Conference, Journal, Preprint, etc.).\n"
                    f"5. VENUE: Extract conference/journal names from headers, first page, or metadata.\n"
                    f"6. GENERAL INSTRUCTIONS:\n"
                    f"   - Use the PDF visual layout to find metadata in margins, headers, footers.\n"
                    f"   - Cross-reference with extracted text for content verification.\n"
                    f"   - Be thorough with author lists, keywords, and citations.\n"
                    f"   - If information is not present, do NOT invent it. Set to null or omit.\n"
                    f"   - Return ONLY a valid, well-formatted JSON object adhering to the schema.\n"
                    f"   - Do NOT include Markdown code blocks. Output raw JSON only."
                )
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_data}"}}
                        ]
                    }
                ]
            
            else:
                print(f"Unknown setup: {setup}")
                return
            
            try:
                # OpenRouter handles response_format="json_object" for supported models
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                )
                predicted_text = response.choices[0].message.content
                
                # Cleanup markdown codeblocks
                predict_text_clean = predicted_text.strip()
                if predict_text_clean.startswith("```json"):
                    predict_text_clean = predict_text_clean[7:-3]
                elif predict_text_clean.startswith("```"):
                    predict_text_clean = predict_text_clean[3:-3]
                predict_text_clean = predict_text_clean.strip()
                
                predicted = json.loads(predict_text_clean)
                predicted_path.write_text(json.dumps(predicted, indent=2), encoding="utf-8")
                print(f"Saved prediction to {predicted_path}")
                
            except Exception as e:
                print(f"Error querying model or parsing JSON for {pdf_path.name}: {e}")
                continue
        else:
            print(f"Loading existing prediction from {predicted_path}...")
            predicted = json.loads(predicted_path.read_text(encoding="utf-8"))
        
        # Evaluate
        print(f"Evaluating {pdf_path.name}...")
        config = ReportConfig(
            output_dir=Path("./eval_results") / f"{setup}_{model_name.replace('/', '_')}_{timestamp}",
            output_name=pdf_path.stem,
            save_json=True,
            save_text=True,
            save_csv=True,
            save_markdown=True,
        )
        builder = ReportBuilder(config)
        report = await builder.build_async(schema, gold, predicted)
        builder.save(report)
        
        overall_stats.append({
            "document": pdf_path.stem,
            "pass_rate": report.overall_pass_rate,
            "score": report.overall_score
        })
        
        print(f"Pass rate: {report.overall_pass_rate:.1%}, Score: {report.overall_score:.3f}")

    # Summary of run
    if overall_stats:
        print(f"\n=== RUN SUMMARY ({setup}) ===")
        print(f"Timestamp: {timestamp}")
        avg_pass = sum(s["pass_rate"] for s in overall_stats) / len(overall_stats)
        avg_score = sum(s["score"] for s in overall_stats) / len(overall_stats)
        print(f"Model: {model_name}")
        print(f"Average Pass Rate: {avg_pass:.1%}")
        print(f"Average Score: {avg_score:.3f}")

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Evaluate extract-bench with different setups.")
    parser.add_argument("--model", type=str, default=os.getenv("OPENROUTER_MODEL", "openrouter/google/gemini-3-flash-preview"),
                        help="The model to use via litellm (e.g. openrouter/anthropic/claude-3-haiku)")
    parser.add_argument("--setup", type=str, choices=["setup1", "setup2", "setup3", "setup4", "setup5", "setup6", "setup7"], required=True,
                        help="setup1: Multimodal PDF + Original Prompt. setup2: Markdown + Original Prompt. setup3: Markdown + Improved Prompt. setup4: Markdown + Metadata-Focused Prompt. setup5: Multimodal PDF + Improved Prompt. setup6: PDF + Markdown + Original Prompt. setup7: PDF + Markdown + Improved Prompt.")
    
    args = parser.parse_args()
    
    asyncio.run(process_and_evaluate(model_name=args.model, setup=args.setup))
