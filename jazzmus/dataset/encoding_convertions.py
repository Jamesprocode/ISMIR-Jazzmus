# Description: This script contains the class GtParser which is used to convert the ground truth
import gin
@gin.configurable
class GtParser:
    def __init__(self, split_enc=False, process_harm=False, character_lvl=False) -> None:
        self.split_enc = split_enc
        self.process_harm = process_harm
        self.character_lvl = character_lvl

    def _get_character_lvl(self, lines):
        # split each character into a token
        tokens = []
        for l in lines:
            for c in l:
                tokens.append(c)

        # if there are < t > or < n > tokens, they should be a single token
        # e.g. < t > should be <t>
        for i, t in enumerate(tokens):
            if t == "<" and tokens[i+1] == "t" and tokens[i+2] == ">":
                tokens[i] = "<t>"
                tokens.pop(i+1)
                tokens.pop(i+1)
            if t == "<" and tokens[i+1] == "n" and tokens[i+2] == ">":
                tokens[i] = "<n>"
                tokens.pop(i+1)
                tokens.pop(i+1)
        return tokens

    def convert(self, src_file: str):
        from jazzmus.dataset.tokenizer import process_text
        with open(src_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            return process_text(lines, self.character_lvl)
        # # read file and get lines
        # # print(f"DEBUG: src_file == {src_file}")
        # reserved_lines = ["!!linebreak", "!!pagebreak", '*I"Voice']
        # tokens = []
        # with open(src_file) as f:
        #     lines = f.read().splitlines()

        #     for l in lines:
        #         # if some part of the line is like any reserved_lines, skip it
        #         if any([rl in l for rl in reserved_lines]):
        #             continue

        #         elements = l.split("\t")

        #         for e in elements:
        #             tokens.append(e)
        #             tokens.append("<t>")

        #         # if the last one is a tab, remove it
        #         if tokens[-1] == "<t>":
        #             tokens.pop()
        #         tokens.append("<n>")
            
        #     if self.character_lvl:
        #         tokens = self._get_character_lvl(tokens)
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
    example_file = "data/jazzmus_dataset_regions/a-day-in-the-life-of-a-fool_version_1_633510.kern"

    gt_parser_raw = GtParser()
    transcript = gt_parser_raw.convert(example_file)
    print("Raw:")
    print(transcript)
    print("Character level:")
    gt_parser_char = GtParser(character_lvl=True)
    transcript = gt_parser_char.convert(example_file)
    print(transcript)
