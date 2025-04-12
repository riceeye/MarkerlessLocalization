from PIL import Image

# Define the image size and grey color
width, height = 512, 512
color = (220, 0, 0)  # RGB values for grey

# Create a new image with the specified color
image = Image.new("RGB", (width, height), color)

# Save the image to a PNG file
image.save("texture.png")

print("Texture file 'cube_texture.png' created successfully.")
