import json

# Set the input and output JSON filenames.
input_file = "annotations.json"   # Your current JSON file
output_file = "new_annotations.json"     # New file in VIA format

# Load the current JSON annotations.
with open(input_file, "r") as f:
    data = json.load(f)

new_data = {}
for filename, annotation in data.items():
    new_annotation = {}
    # Preserve filename and size
    new_annotation["filename"] = annotation.get("filename", filename)
    new_annotation["size"] = annotation.get("size", 0)

    # Convert the "regions" field: if it's a list, make it a dictionary.
    regions = annotation.get("regions", [])
    new_regions = {}
    if isinstance(regions, list):
        for idx, region in enumerate(regions):
            new_regions[str(idx)] = region
    elif isinstance(regions, dict):
        new_regions = regions
    new_annotation["regions"] = new_regions

    # Keep file attributes.
    new_annotation["file_attributes"] = annotation.get("file_attributes", {})

    new_data[filename] = new_annotation

# Save the new VIA-formatted JSON file.
with open(output_file, "w") as f:
    json.dump(new_data, f, indent=4)

print(f"Converted annotations saved to {output_file}")
