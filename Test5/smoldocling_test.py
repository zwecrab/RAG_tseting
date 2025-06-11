#!/usr/bin/env python3
"""
SmolDocling Table Extractor - Extract tables in OTSL format from document images
Requires: pip install torch transformers docling_core pillow
"""

import torch
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image


@dataclass
class TableExtraction:
    """Container for extracted table data"""
    otsl_content: str
    raw_doctags: str
    table_index: int
    location_info: Optional[Dict] = None


class SmolDoclingTableExtractor:
    """SmolDocling-based table extractor with OTSL format output"""
    
    def __init__(self, model_name: str = "ds4sd/SmolDocling-256M-preview"):
        """Initialize the SmolDocling model"""
        print("=" * 60)
        print("INITIALIZING SMOLDOCLING TABLE EXTRACTOR")
        print("=" * 60)
        
        print("Step 1: Detecting compute device...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Using device: {self.device}")
        
        print(f"\nStep 2: Loading SmolDocling processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("✓ Processor loaded successfully")
        
        print(f"\nStep 3: Loading SmolDocling model...")
        print(f"  - Model: {model_name}")
        print(f"  - Precision: bfloat16")
        
        # Determine attention implementation with fallback
        attention_impl = self._get_attention_implementation()
        print(f"  - Attention: {attention_impl}")
        
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                _attn_implementation=attention_impl,
            ).to(self.device)
            print("✓ Model loaded and moved to device successfully")
        except Exception as e:
            print(f"❌ Error loading model with {attention_impl}: {str(e)}")
            print("  - Trying fallback with 'eager' attention...")
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager",
                ).to(self.device)
                print("✓ Model loaded successfully with 'eager' attention")
            except Exception as e2:
                print(f"❌ Failed with eager attention: {str(e2)}")
                print("  - Trying without attention specification...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                ).to(self.device)
                print("✓ Model loaded successfully with default attention")
        
        print("✓ Initialization complete!\n")
    
    def _get_attention_implementation(self) -> str:
        """Determine the best attention implementation based on available packages"""
        if self.device == "cpu":
            return "eager"
        
        # Check if flash_attn is available
        try:
            import flash_attn
            print("  - flash_attn package detected")
            return "flash_attention_2"
        except ImportError:
            print("  - flash_attn package not installed, using 'eager' attention")
            return "eager"
    
    def extract_tables_from_image(self, image_path: str, prompt: str = "Convert table to OTSL.") -> List[TableExtraction]:
        """
        Extract tables from an image and return OTSL format
        
        Supported formats: PNG, JPG, JPEG, and URLs
        
        Args:
            image_path: Path to PNG/JPG/JPEG image file or URL
            prompt: Instruction prompt for the model
            
        Returns:
            List of TableExtraction objects containing OTSL data
        """
        print("\n" + "=" * 60)
        print("EXTRACTING TABLES FROM IMAGE")
        print("=" * 60)
        
        try:
            print(f"Step 1: Loading image from: {image_path}")
            print(f"  - Prompt: '{prompt}'")
            
            # Load image with format detection
            if isinstance(image_path, str):
                if image_path.startswith('http'):
                    print("  - Loading from URL...")
                    image = load_image(image_path)
                    print("  ✓ URL image loaded successfully")
                else:
                    print("  - Loading from local file...")
                    # Check if file exists and format
                    if not Path(image_path).exists():
                        raise FileNotFoundError(f"Image file not found: {image_path}")
                    
                    file_ext = Path(image_path).suffix.lower()
                    if file_ext not in ['.png', '.jpg', '.jpeg']:
                        print(f"  ⚠ Warning: File extension '{file_ext}' - expected .png, .jpg, or .jpeg")
                    
                    image = Image.open(image_path).convert('RGB')
                    print(f"  ✓ Local image loaded successfully (format: {image.format}, size: {image.size})")
            else:
                print("  - Using provided image object...")
                image = image_path
                print("  ✓ Image object accepted")
            
            print(f"\nStep 2: Preparing input messages...")
            # Create input messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            print("  ✓ Messages prepared")
            
            print(f"\nStep 3: Applying chat template...")
            # Prepare inputs
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            print(f"  ✓ Chat template applied (length: {len(prompt_text)} chars)")
            
            print(f"\nStep 4: Processing image and text through processor...")
            inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)
            print(f"  ✓ Inputs processed and moved to {self.device}")
            print(f"  - Input IDs shape: {inputs.input_ids.shape}")
            print(f"  - Pixel values shape: {inputs.pixel_values.shape}")
            
            print(f"\nStep 5: Generating model output...")
            print("  - This may take 10-30 seconds depending on image complexity...")
            # Generate output
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            print("  ✓ Model generation completed")
            
            print(f"\nStep 6: Decoding output...")
            # Decode output
            prompt_length = inputs.input_ids.shape[1]
            trimmed_generated_ids = generated_ids[:, prompt_length:]
            doctags = self.processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=False,
            )[0].lstrip()
            
            print(f"  ✓ Output decoded (length: {len(doctags)} chars)")
            print(f"  - First 100 chars: {doctags[:100]}...")
            
            print(f"\nStep 7: Extracting tables from DocTags...")
            # Extract tables from DocTags
            tables = self._extract_tables_from_doctags(doctags)
            
            print(f"  ✓ Found {len(tables)} table(s)")
            
            return tables
            
        except Exception as e:
            print(f"  ❌ Error processing image: {str(e)}")
            return []
    
    def _extract_tables_from_doctags(self, doctags: str) -> List[TableExtraction]:
        """Extract OTSL table content from DocTags output"""
        print(f"\nStep 7a: Parsing DocTags for OTSL tables...")
        tables = []
        
        # Find all OTSL table blocks using regex
        print(f"  - Searching for <otsl> tags in DocTags output...")
        otsl_pattern = r'<otsl>(.*?)</otsl>'
        matches = re.finditer(otsl_pattern, doctags, re.DOTALL)
        
        matches_list = list(matches)
        print(f"  ✓ Found {len(matches_list)} OTSL block(s)")
        
        if len(matches_list) == 0:
            print("  - No OTSL blocks found. Checking for alternative table formats...")
            # Check if there are any table-related tags
            if '<table>' in doctags or '<tr>' in doctags:
                print("  ⚠ Found HTML table tags instead of OTSL")
            elif 'table' in doctags.lower():
                print("  ⚠ Found text mentioning 'table' but no structured format")
            else:
                print("  - No table content detected in the output")
        
        for i, match in enumerate(matches_list):
            print(f"\n  Processing table {i+1}:")
            otsl_content = match.group(1).strip()
            print(f"    - OTSL content length: {len(otsl_content)} characters")
            print(f"    - Preview: {otsl_content[:100]}..." if len(otsl_content) > 100 else f"    - Content: {otsl_content}")
            
            # Try to extract location information if present
            print(f"    - Extracting location information...")
            location_info = self._extract_location_info(doctags, match.start(), match.end())
            if location_info:
                print(f"    ✓ Location found: {location_info}")
            else:
                print(f"    - No location information found")
            
            table = TableExtraction(
                otsl_content=otsl_content,
                raw_doctags=match.group(0),
                table_index=i,
                location_info=location_info
            )
            tables.append(table)
            print(f"    ✓ Table {i+1} extracted successfully")
        
        return tables
    
    def _extract_location_info(self, doctags: str, start_pos: int, end_pos: int) -> Optional[Dict]:
        """Extract location tags around the OTSL content"""
        # Look for location tags before and after the OTSL block
        context = doctags[max(0, start_pos-200):min(len(doctags), end_pos+200)]
        
        # Find location tags pattern
        loc_pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        loc_match = re.search(loc_pattern, context)
        
        if loc_match:
            return {
                'x1': int(loc_match.group(1)),
                'y1': int(loc_match.group(2)),
                'x2': int(loc_match.group(3)),
                'y2': int(loc_match.group(4))
            }
        return None
    
    def convert_to_docling_document(self, image_path: str, tables: List[TableExtraction]) -> DoclingDocument:
        """Convert extracted tables to a DoclingDocument for further processing"""
        try:
            # Load image for DocTags document creation
            if isinstance(image_path, str) and not image_path.startswith('http'):
                image = Image.open(image_path).convert('RGB')
            else:
                image = load_image(image_path)
            
            # Create a minimal DocTags structure with tables
            full_doctags = "<doctag>"
            for table in tables:
                full_doctags += table.raw_doctags
            full_doctags += "</doctag>"
            
            # Create DocTags document
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([full_doctags], [image])
            
            # Create DoclingDocument
            doc = DoclingDocument(name="Extracted Tables")
            doc.load_from_doctags(doctags_doc)
            
            return doc
            
        except Exception as e:
            print(f"Error creating DoclingDocument: {str(e)}")
            return None
    
    def save_tables_to_files(self, tables: List[TableExtraction], output_dir: str = "output"):
        """Save extracted tables to individual files"""
        print(f"\n" + "=" * 60)
        print("SAVING EXTRACTED TABLES")
        print("=" * 60)
        
        print(f"Step 1: Creating output directory: {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"✓ Directory created/confirmed: {output_path.absolute()}")
        
        if not tables:
            print("⚠ No tables to save")
            return
        
        print(f"\nStep 2: Saving {len(tables)} table(s) to files...")
        
        for i, table in enumerate(tables):
            print(f"\n  Saving table {i+1}:")
            
            # Save OTSL content
            otsl_file = output_path / f"table_{i+1}.otsl"
            print(f"    - Writing OTSL to: {otsl_file}")
            with open(otsl_file, 'w', encoding='utf-8') as f:
                f.write(table.otsl_content)
            print(f"    ✓ OTSL file saved ({len(table.otsl_content)} chars)")
            
            # Save metadata
            metadata = {
                'table_index': table.table_index,
                'location_info': table.location_info,
                'otsl_length': len(table.otsl_content),
                'raw_doctags_length': len(table.raw_doctags)
            }
            
            metadata_file = output_path / f"table_{i+1}_metadata.json"
            print(f"    - Writing metadata to: {metadata_file}")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"    ✓ Metadata file saved")
            
            # Save raw DocTags for debugging
            doctags_file = output_path / f"table_{i+1}_raw_doctags.txt"
            print(f"    - Writing raw DocTags to: {doctags_file}")
            with open(doctags_file, 'w', encoding='utf-8') as f:
                f.write(table.raw_doctags)
            print(f"    ✓ Raw DocTags file saved")
        
        print(f"\n✓ All files saved to: {output_path.absolute()}")


def main():
    """Example usage of the SmolDocling Table Extractor"""
    
    print("=" * 80)
    print("SMOLDOCLING TABLE EXTRACTOR - DEMO")
    print("=" * 80)
    print("Supported input formats: PNG, JPG, JPEG files and URLs")
    print("Output format: OTSL (Optimized Table Structure Language)")
    print()
    
    # Initialize extractor
    extractor = SmolDoclingTableExtractor()
    
    # Example image with table (you can replace with your image path)
    # Using a sample image URL for demonstration
    image_url = "https://upload.wikimedia.org/wikipedia/commons/7/76/GazettedeFrance.jpg"
    
    # You can also use local files:
    # image_path = "path/to/your/table_image.png"
    # image_path = "path/to/your/table_image.jpg"
    # image_path = "path/to/your/table_image.jpeg"
    
    print(f"Demo input: {image_url}")
    print("NOTE: Replace with your own PNG/JPG/JPEG file path for real usage")
    
    # Extract tables using different prompts
    prompts = [
        "Convert table to OTSL.",
        "Extract all tables from this document.",
        "Convert this page to docling."
    ]
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"TRYING PROMPT {prompt_idx + 1}/{len(prompts)}: '{prompt}'")
        print(f"{'='*80}")
        
        tables = extractor.extract_tables_from_image(image_url, prompt)
        
        if tables:
            print(f"\n🎉 SUCCESS! Found {len(tables)} table(s)")
            
            for i, table in enumerate(tables):
                print(f"\n📋 TABLE {i+1} SUMMARY:")
                print(f"  - OTSL Content Length: {len(table.otsl_content)} characters")
                print(f"  - Location: {table.location_info}")
                print(f"  - Preview: {table.otsl_content[:200]}...")  # First 200 chars
                
            # Save tables to files
            output_dir = f"output_{prompt.replace(' ', '_').replace('.', '').replace(',', '')}"
            extractor.save_tables_to_files(tables, output_dir)
            
            print(f"\n" + "=" * 60)
            print("CONVERTING TO DOCLING DOCUMENT")
            print("=" * 60)
            
            # Convert to DoclingDocument for further processing
            print("Step 1: Converting to DoclingDocument...")
            doc = extractor.convert_to_docling_document(image_url, tables)
            if doc:
                print("✓ DoclingDocument created successfully!")
                
                print("\nStep 2: Exporting to Markdown...")
                # Export to Markdown
                markdown_output = doc.export_to_markdown()
                markdown_file = f"{output_dir}/document.md"
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_output)
                print(f"✓ Exported to: {markdown_file}")
                print(f"  - Content length: {len(markdown_output)} characters")
            else:
                print("❌ Failed to create DoclingDocument")
                
            break  # Use the first successful prompt
        else:
            print(f"\n❌ No tables found with prompt: '{prompt}'")
            if prompt_idx < len(prompts) - 1:
                print("Trying next prompt...")
    
    print(f"\n{'='*80}")
    print("DEMO COMPLETED")
    print("='*80}")
    print("Check the output_* directories for extracted table files!")


if __name__ == "__main__":
    main()