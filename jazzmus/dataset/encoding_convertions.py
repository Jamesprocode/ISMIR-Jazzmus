# Description: This script contains the class GtParser which is used to convert the ground truth
class GtParser:
    def __init__(self, split_enc=False, process_harm=False) -> None:
        self.split_enc = split_enc
        self.process_harm = process_harm

    def convert(self, src_file: str):
        # read file and get lines
        # print(f"DEBUG: src_file == {src_file}")
        tokens = []
        with open(src_file) as f:
            lines = f.read().splitlines()

            # if self.split_enc:
            #     lines = self._split_encode(lines)

            # if self.process_harm:
            #     lines = self._harm_split_encode(lines)
            for l in lines:
                if "!!linebreak" in l:
                    continue

                elements = l.split("\t")

                for e in elements:
                    tokens.append(e)
                    tokens.append("<t>")

                # if the last one is a tab, remove it
                if tokens[-1] == "<t>":
                    tokens.pop()
                tokens.append("<n>")
        return tokens

    def _split_encode(self, lines):
        transcript = []
        for line in lines:
            symbol, position = line.split(":")[0], line.split(":")[1]
            transcript.append(symbol)
            transcript.append("<pos>" + position)
        return transcript

    def _extract_chord_root(self, chord_info):
        if chord_info[1] == "b" or chord_info[1] == "#":
            # append to previous token
            root = chord_info[0:2]
            chord_info = chord_info[2:]
        else:
            root = chord_info[0]
            chord_info = chord_info[1:]
        return root, chord_info

    def _harm_split_encode(self, lines):
        transcript = []
        for line in lines:
            if "harm" in line:
                # we get the actual chord. e.g. harm.C#m7
                # root is C# and chord_info is m7
                chord = line.split(".")[1]
                if len(chord) == 1:
                    transcript.append(chord)
                    continue
                root, chord_info = self._extract_chord_root(chord)

                transcript.append("chord.root." + root)
                transcript.append("chord.shorthand." + chord_info)
            else:
                transcript.append(line)

        return transcript


if __name__ == "__main__":
    example_file = "data/jazzmus/gt/img_0_0.txt"

    gt_parser_raw = GtParser()
    transcript = gt_parser_raw.convert(example_file)
    print("Raw:")
    print(transcript)
    print()

    gt_parser_split = GtParser(split_enc=True)
    transcript = gt_parser_split.convert(example_file)
    print("Split:")
    print(transcript)
    print()

    # gt_parser_harm_proc = GtParser(process_harm=True)
    # transcript = gt_parser_harm_proc.convert(example_file)
    # print("Harm proc:")
    # print(transcript)
    # print()

    gt_parser_harm_split = GtParser(process_harm=True, split_enc=True)
    transcript = gt_parser_harm_split.convert(example_file)
    print("Harm proc and split:")
    print(transcript)
