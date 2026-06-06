import json

# Load the ipynb file and convert its code and markdown cells to a readable txt format
with open('Image_Classification (2).ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

output_lines = []

# Title banner
output_lines.append("=" * 80)
output_lines.append("JUPYTER NOTEBOOK EXPORT (.TXT)")
output_lines.append(f"File Name: Image_Classification (2).ipynb")
output_lines.append("=" * 80 + "\n")

for i, cell in enumerate(notebook.get('cells', [])):
    cell_type = cell.get('cell_type', '')
    source = "".join(cell.get('source', []))
    
    if cell_type == 'markdown':
        output_lines.append(f"--- [Cell {i+1}: Markdown] ---")
        output_lines.append(source.strip())
        output_lines.append("\n" + "-" * 40 + "\n")
    elif cell_type == 'code':
        output_lines.append(f"--- [Cell {i+1}: Code] ---")
        output_lines.append(source.rstrip())
        output_lines.append("\n" + "-" * 40 + "\n")

# Save as a plain text file
output_file_path = 'Image_Classification_2.txt'
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

print(f"File saved successfully to {output_file_path}")
