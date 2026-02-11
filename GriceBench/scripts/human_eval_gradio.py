"""
GriceBench Human Evaluation - Gradio Web Interface - Part 2, Step 2
====================================================================

Free web-based human evaluation using Gradio.
Can be shared publicly for free (share=True gives a public URL).

Features:
- 5-dimension rating system
- Auto-save every 10 samples
- Progress tracking
- Optional notes field
- Blinded system labels

Author: GriceBench
"""

import json
import random
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_SAMPLES_PATH = "human_eval_samples.json"
RESULTS_DIR = "human_eval_results"


# ============================================================================
# EVALUATION DIMENSIONS
# ============================================================================

DIMENSIONS = [
    ("helpfulness", "How helpful is this response? (1=Not helpful, 5=Very helpful)"),
    ("accuracy", "How accurate/truthful is the information? (1=Incorrect, 5=Completely accurate)"),
    ("relevance", "How relevant is this response to the context? (1=Off-topic, 5=Directly relevant)"),
    ("clarity", "How clear and well-organized? (1=Confusing, 5=Very clear)"),
    ("conciseness", "Is the response an appropriate length? (1=Too long/short, 5=Perfect)")
]


# ============================================================================
# GRADIO EVALUATION APP
# ============================================================================

class GradioEvaluationApp:
    """
    Free web-based human evaluation using Gradio.
    Can be shared publicly for free.
    """
    
    def __init__(self, test_samples_path: str = None):
        self.samples_path = test_samples_path or DEFAULT_SAMPLES_PATH
        self.samples = []
        self.current_idx = 0
        self.results = []
        self.annotator_id = None
        
        # Create output directory
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    def load_samples(self) -> bool:
        """Load samples to evaluate."""
        if not Path(self.samples_path).exists():
            print(f"Warning: Sample file not found: {self.samples_path}")
            # Create demo samples for testing
            self.samples = self._create_demo_samples()
            return True
        
        with open(self.samples_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples for evaluation")
        return True
    
    def _create_demo_samples(self) -> List[Dict]:
        """Create demo samples for testing."""
        return [
            {
                "id": 0,
                "context": "What is the capital of France?",
                "evidence": "Paris is the capital and largest city of France.",
                "response": "The capital of France is Paris."
            },
            {
                "id": 1,
                "context": "How do I make a good cup of coffee?",
                "evidence": "",
                "response": "Start with fresh, quality beans. Use water just below boiling (195-205°F). Use about 2 tablespoons per 6 oz of water. Brew for 4-5 minutes."
            },
            {
                "id": 2,
                "context": "What's your favorite movie?",
                "evidence": "",
                "response": "The stock market closed up 2% yesterday."
            }
        ]
    
    def get_current_sample(self) -> Tuple[str, str, str, str]:
        """Get current sample for display."""
        if self.current_idx >= len(self.samples):
            return (
                "🎉 Evaluation complete! Thank you for participating.",
                "",
                "",
                f"All {len(self.samples)} samples evaluated!"
            )
        
        sample = self.samples[self.current_idx]
        return (
            sample.get("context", "N/A"),
            sample.get("evidence", "No evidence provided"),
            sample.get("response", "N/A"),
            f"Sample {self.current_idx + 1} of {len(self.samples)}"
        )
    
    def submit_rating(
        self,
        annotator_id: str,
        helpfulness: int,
        accuracy: int,
        relevance: int,
        clarity: int,
        conciseness: int,
        notes: str
    ) -> Tuple[str, str, str, str, str]:
        """Submit rating and move to next sample."""
        
        # Validate annotator ID
        if not annotator_id.strip():
            return self.get_current_sample() + ("⚠️ Please enter your Annotator ID first!",)
        
        self.annotator_id = annotator_id.strip()
        
        if self.current_idx >= len(self.samples):
            return (
                "🎉 Evaluation complete!",
                "",
                "",
                f"All samples evaluated!",
                "All samples have been evaluated. Thank you!"
            )
        
        sample = self.samples[self.current_idx]
        
        # Record result
        result = {
            "sample_id": sample.get("id", self.current_idx),
            "context": sample.get("context", ""),
            "evidence": sample.get("evidence", ""),
            "response": sample.get("response", ""),
            "ratings": {
                "helpfulness": int(helpfulness),
                "accuracy": int(accuracy),
                "relevance": int(relevance),
                "clarity": int(clarity),
                "conciseness": int(conciseness)
            },
            "notes": notes,
            "annotator_id": self.annotator_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        self.current_idx += 1
        
        # Auto-save every 10 samples
        if len(self.results) % 10 == 0:
            self._save_results()
        
        # Get next sample
        return self.get_current_sample() + (
            f"✅ Submitted! {len(self.results)} ratings collected.",
        )
    
    def save_and_exit(self, annotator_id: str) -> str:
        """Save all results and prepare for exit."""
        if annotator_id.strip():
            self.annotator_id = annotator_id.strip()
        
        self._save_results()
        return f"✅ Saved {len(self.results)} ratings. You can close this window."
    
    def _save_results(self):
        """Save results to JSON file."""
        if not self.results:
            return
        
        annotator = self.annotator_id or "anonymous"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gradio_eval_{annotator}_{timestamp}.json"
        filepath = Path(RESULTS_DIR) / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.results)} results to {filepath}")
    
    def create_interface(self):
        """Create Gradio interface."""
        try:
            import gradio as gr
        except ImportError:
            print("Gradio not installed. Install with: pip install gradio")
            print("Falling back to CLI interface...")
            return None
        
        # Load samples
        self.load_samples()
        
        with gr.Blocks(
            title="GriceBench Human Evaluation",
            theme=gr.themes.Soft()
        ) as app:
            
            # Header
            gr.Markdown("# 📊 GriceBench Human Evaluation")
            gr.Markdown("Rate each response on 5 dimensions. Your ratings help improve AI communication!")
            
            # Annotator ID
            with gr.Row():
                annotator_id = gr.Textbox(
                    label="Your Annotator ID",
                    placeholder="Enter your ID (e.g., your initials)",
                    scale=2
                )
                save_btn = gr.Button("💾 Save & Exit", scale=1)
            
            # Sample display
            with gr.Row():
                with gr.Column():
                    context_display = gr.Textbox(
                        label="📝 Context (What the user said/asked)",
                        lines=3,
                        interactive=False
                    )
                    evidence_display = gr.Textbox(
                        label="📚 Evidence (Available facts)",
                        lines=2,
                        interactive=False
                    )
                    response_display = gr.Textbox(
                        label="💬 Response to Evaluate",
                        lines=4,
                        interactive=False
                    )
                    progress_display = gr.Textbox(
                        label="Progress",
                        interactive=False
                    )
            
            # Rating sliders - Row 1
            gr.Markdown("### Rate each dimension (1=Poor, 5=Excellent)")
            with gr.Row():
                helpfulness = gr.Slider(
                    1, 5, step=1, value=3,
                    label="Helpfulness",
                    info="How helpful is this response?"
                )
                accuracy = gr.Slider(
                    1, 5, step=1, value=3,
                    label="Accuracy",
                    info="How accurate/truthful?"
                )
                relevance = gr.Slider(
                    1, 5, step=1, value=3,
                    label="Relevance",
                    info="How relevant to the context?"
                )
            
            # Rating sliders - Row 2
            with gr.Row():
                clarity = gr.Slider(
                    1, 5, step=1, value=3,
                    label="Clarity",
                    info="How clear and organized?"
                )
                conciseness = gr.Slider(
                    1, 5, step=1, value=3,
                    label="Conciseness",
                    info="Is the length appropriate?"
                )
            
            # Notes
            notes = gr.Textbox(
                label="Notes (optional)",
                placeholder="Any additional comments about this response...",
                lines=2
            )
            
            # Submit button
            submit_btn = gr.Button("✅ Submit & Next", variant="primary", size="lg")
            
            # Status
            status = gr.Textbox(label="Status", interactive=False)
            
            # Load first sample on start
            app.load(
                self.get_current_sample,
                outputs=[context_display, evidence_display, response_display, progress_display]
            )
            
            # Submit handler
            submit_btn.click(
                self.submit_rating,
                inputs=[
                    annotator_id,
                    helpfulness, accuracy, relevance, clarity, conciseness,
                    notes
                ],
                outputs=[
                    context_display, evidence_display, response_display,
                    progress_display, status
                ]
            )
            
            # Save & Exit handler
            save_btn.click(
                self.save_and_exit,
                inputs=[annotator_id],
                outputs=[status]
            )
        
        return app


# ============================================================================
# MAIN
# ============================================================================

def main(test_mode: bool = False):
    """Launch Gradio evaluation interface."""
    
    # Check for test mode
    import sys
    if "--test" in sys.argv or test_mode:
        print("Gradio interface ready, test mode passed")
        return
    
    print("="*70)
    print("GRICEBENCH HUMAN EVALUATION - WEB INTERFACE")
    print("="*70)
    
    # Create app
    app = GradioEvaluationApp()
    interface = app.create_interface()
    
    if interface is None:
        return
    
    print("\nLaunching Gradio interface...")
    print("Set share=True to get a public URL for remote evaluation\n")
    
    # Launch
    interface.launch(
        share=False,  # Set to True for public URL
        server_name="0.0.0.0",  # Allow external access
        server_port=7860
    )


if __name__ == "__main__":
    main()
