import json
import os
import time
import tqdm
import google.generativeai as genai
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'dpo_datacleaning_outcomes/scored_data.json'
OUTPUT_FILE = 'dpo_datacleaning_outcomes/synthetic_candidates.json'
API_KEY = os.environ.get("GEMINI_API_KEY") 

# If no env var, ask user input
if not API_KEY:
    print("⚠️ GEMINI_API_KEY not found in environment variables.")
    API_KEY = input("Please enter your Gemini API Key: ").strip()

genai.configure(api_key=API_KEY)

# Generation Config
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

# The Strict Alignment System Instruction
SYSTEM_INSTRUCTION = """You are a Gricean Cooperative Assistant. 
Your task is to generate responses that strictly adhere to all four Gricean Maxims:
1. Quantity: Be as informative as required, but no more.
2. Quality: Do not say what you believe to be false or lack evidence for.
3. Relation: Be strictly relevant to the user's prompt.
4. Manner: Be perspicuous—avoid obscurity, ambiguity, and unnecessary verbosity. Be orderly and polite.

Context: You are providing a 'chosen' response for a DPO dataset. 
Your output must be significantly better than a typical chatbot response in terms of cooperation and clarity.
Do not be chatty. Do not offer unsolicited advice. Answer the prompt directly and cooperatively."""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=SYSTEM_INSTRUCTION
)

# ==========================================
# MAIN LOGIC
# ==========================================

def get_candidates_for_generation(data):
    """Select prompts where the original response failed our strict filter."""
    candidates = []
    print("Selecting prompts...")
    
    for entry in data:
        margins = entry.get('margins', {})
        qty = margins.get('quantity', 0)
        qlt = margins.get('quality', 0)
        rel = margins.get('relation', 0)
        man = margins.get('manner', 0)
        
        # Condition: Failed the strict filter (at least one <= 0)
        # Optimization: Prioritize cases where Relation is good (>0) but others failed, 
        # as these are relevant prompts that just had bad responses.
        if not (qty > 0 and qlt > 0 and rel > 0 and man > 0):
            # We need the prompt and the original rejected response (to form the pair later)
            # Actually, we keep the entire entry structure and just overwrite 'chosen' later
            candidates.append(entry)
            
    print(f"Select {len(candidates)} prompts for synthetic generation out of {len(data)} total.")
    return candidates

def generate_responses():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    targets = get_candidates_for_generation(data)
    
    # Check for existing progress
    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Resuming... Found {len(results)} already generated.")
        except json.JSONDecodeError:
            print("Warning: Output file corrupt, starting fresh.")
    
    # Create valid lookup set for skip logic
    processed_prompts = {r['prompt'] for r in results}
    
    # Filter targets to remove already processed
    todos = [t for t in targets if t['prompt'] not in processed_prompts]
    print(f"Remaining to generate: {len(todos)}")
    
    # Rate limit handling (simple sleep)
    # Gemini Flash has high limits, but let's be safe: ~15 RPM free tier, higher on paid.
    # Assuming standard tier: 15 RPM = 4 seconds/req. 
    # If user has paid tier, they can remove the sleep.
    SLEEP_SEC = 2.0 

    print("Starting generation... (Press Ctrl+C to stop safely)")
    
    try:
        for i, item in enumerate(tqdm.tqdm(todos)):
            prompt_text = item['prompt']
            
            # Extract just the dialogue history if possible, or use full prompt
            # The prompt field often contains "Context: ... Evidence: ... Generate a cooperative response:"
            # We pass the whole thing to Gemini as the user message.
            
            try:
                chat = model.start_chat(history=[])
                response = chat.send_message(prompt_text)
                synthetic_response = response.text.strip()
                
                # Construct result entry
                new_entry = item.copy()
                new_entry['synthetic_chosen'] = synthetic_response
                # We rename original 'chosen' to 'original_chosen_failed' for record keeping
                new_entry['original_chosen_failed'] = item['chosen']
                new_entry['chosen'] = synthetic_response # This becomes the new chosen
                
                results.append(new_entry)
                
                # Save every 10 items
                if i % 10 == 0:
                     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                
                time.sleep(SLEEP_SEC)
                
            except Exception as e:
                print(f"\n❌ Error on item {i}: {e}")
                print("Waiting 30s before retry...")
                time.sleep(30) # Backoff for quotas
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user. Saving progress...")
    finally:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved {len(results)} synthetic pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_responses()
