from pathlib import Path
from jazzmus.utils.la_inference import run_la_inference

# Convert SVG to PNG
import io
from cairosvg import svg2png
from PIL import Image

# read the available SVG files
svg_files = list(Path("./data/jazzmus_dataset_synthetic").glob("*.svg"))

# convert each SVG file to PNG
for svgfile in svg_files:
    pngfile = str(svgfile.with_suffix(".png"))

    with open(svgfile, mode="r") as input_file:
        svg_stream = input_file.read()
        # See https://github.com/Kozea/CairoSVG/issues/300 for the reason why we have to replace inherit with visible here
        svg_stream = svg_stream.replace("overflow=\"inherit\"", "overflow=\"visible\"")
        svg2png(bytestring=svg_stream, background_color="transparent", write_to=pngfile)
            

png_files = list(Path("./data/jazzmus_dataset_synthetic").glob("*.png"))

run_la_inference(png_files, "./data/la_result")
