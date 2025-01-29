import os


class GtParser:
    def __init__(self) -> None:
        self.debug_file = "debug/missing_gt_files.dat"

        if os.path.exists(self.debug_file):
            print("Removing previous gt debug file: ", self.debug_file)
            os.remove(self.debug_file)

    def convert(self, src_file: str):
        # read file and get lines
        try:
            with open(src_file) as f:
                lines = f.read().splitlines()

        except Exception:
            with open(self.debug_file, "a") as f:
                f.write(f"{src_file}\n")
            lines = []
        return lines
