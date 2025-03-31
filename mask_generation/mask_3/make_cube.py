from PIL import Image

# Define the image size and grey color
width, height = 512, 512
grey_color = (128, 128, 128)  # RGB values for grey

# Create a new image with the specified color
image = Image.new("RGB", (width, height), grey_color)

# Save the image to a PNG file
image.save("arm_texture.png")

print("Texture file 'cube_texture.png' created successfully.")
