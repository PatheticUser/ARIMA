import nbformat

files = ["04_modeling_and_training.ipynb", "main.ipynb"]

merged = nbformat.v4.new_notebook()
merged.cells = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        merged.cells.extend(nb.cells)

with open("merged.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(merged, f)

print("Merged notebook saved as merged.ipynb")
