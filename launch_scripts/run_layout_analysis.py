from pathlib import Path
from jazzmus.utils.la_inference import run_la_inference

files = list(Path("./data/jazzmus_dataset").glob("*.png"))

files = [
    "/Users/jc/Documents/tesis/ISMIR-Jazzmus/la_result/jazzmus_1.jpeg",
    "/Users/jc/Documents/tesis/ISMIR-Jazzmus/la_result/jazzmus_2.jpeg",
]

run_la_inference(files, "la_result")
