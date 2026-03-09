from PIL import Image, ImageDraw, ImageFont

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for letter in letters:
    # Create black background image (28x28)
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)

    # Try loading Arial font (optional)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # NEW METHOD (instead of textsize)
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center position
    position = ((28 - text_width) // 2, (28 - text_height) // 2)

    # Draw white letter
    draw.text(position, letter, fill=255, font=font)

    # Save image
    img.save(f"{letter}.png")

print("✅ A–Z images created successfully!")