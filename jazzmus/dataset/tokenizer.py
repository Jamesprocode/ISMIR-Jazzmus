
def process_text(lines, char_lvl: bool = False, medium_lvl: bool = False):
    """Reads and processes the input file with text preprocessing and optional character-level or middle level tokenization."""

    reserved_lines = {"!!linebreak", "!!pagebreak", "*I", "*F:", "!LO"}
    piece_started = False
    tokens = []

    for line in lines:
        # remove the X character as it does not have graphical impact
        # TODO remove this and modify the dataset
        line = line.replace("X", "")
        if line[0].isdigit() or "=" in line:
            piece_started = True

        # Skip reserved lines
        if any(reserved in line for reserved in reserved_lines):
            continue

        if medium_lvl:
            tokens.extend(middle_level_split(line.replace("\n", ""), piece_started))
        else:
            line_elements = line.replace("\n", "").split("\t")

            if char_lvl:
                for element in line_elements:
                    for char in element:
                        tokens.append(char)
                    tokens.append("<t>")
            else:
                for element in line_elements:
                    tokens.append(element)
                    tokens.append("<t>")

            # Remove the last tab token
            if tokens[-1] == "<t>":
                tokens.pop()
            tokens.append("<n>")

    return tokens

def middle_level_split(line, piece_started):
    # handle non note-chord lines
    if not piece_started or "=" in line:
        elements = line.split("\t")
        tokens = []
        for element in elements:
            tokens.append(element)
            tokens.append("<t>")
        if tokens[-1] == "<t>":
            tokens.pop()
        tokens.append("<n>")
    else:
        # last token from line.split("\t") is the chord, the rest are notes
        tokens = line.split("\t")
        if len(tokens) == 1:
            # single spline, only notes
            tokens = note_split(tokens[0])
            tokens.append("<n>")
        else:
            notes = tokens[:-1]
            chord = tokens[-1]
            tokens = []
            for note in notes:
                tokens.extend(note_split(note))
                tokens.append("<t>")
            tokens.extend(chord_split(chord))
            tokens.append("<n>")
    return tokens


def note_split(note_string):
    if note_string in [".", "*v", "*^", "*"]:
        tokens = [note_string]
    else:
        tokens = []
        # group in dedicated symbols every pitch letter, i.e., from [a,b,c,d,e,f,g], upper or lower case
        # the rest is tokenized character by character
        pitch_letters = set('abcdefgABCDEFG')
        # tokens = note_string.split()
        # if two consecutive characters are the same, add them to tokens together, otherwise add them separately
        current_pitch = ""
        for i,char in enumerate(note_string):
            if char in pitch_letters:
                current_pitch = current_pitch + char
            else:
                if current_pitch:
                    tokens.append(current_pitch)
                    current_pitch = ""
                tokens.append(char)
        if current_pitch: # if the string ends with a pitch letter
            tokens.append(current_pitch)
    return tokens

def chord_split(chord_string):
    if chord_string in [".", "*v", "*^", "*"]:
       tokens = [chord_string]
    else:
        # split chord string into root, type, a list of extensions, and bass
        root_string = chord_string.split(":")[0]
        root = []
        root.extend(process_chord_pitch(root_string))
        # separate # and - from root
        
        bass = chord_string.split("/")[1] if "/" in chord_string else "none"
        if len(chord_string.split(":"))==1: # chord without any extension
            # TODO : remove this when the new dataset is ready
            chord_type = "none"
        else:
            chord_type = chord_string.split(":")[1] if "/" not in chord_string else chord_string.split(":")[1].split("/")[0]
        # elements between parentheses in chord_types are extensions modifiers
        if "(" in chord_type:
            extensions_full = chord_type.split("(")[1].split(")")[0].split(",")
        else:
            extensions_full = []
        extensions_single = []
        for ext in extensions_full:
            # further split extensions if they have # or - in them
            if "#" in ext:
                extensions_single.append("#")
                extensions_single.append("<chord-extension>" + ext.replace("#",""))
            elif "b" in ext:
                extensions_single.append("b")
                extensions_single.append("<chord-extension>" + ext.replace("b",""))
            else:
                extensions_single.append("<chord-extension>" + ext)
            extensions_single.append(",")
        # remove the last comma
        if len(extensions_single) !=0:
            extensions_single.pop()
        # now remove extensions from chord_types
        chord_type = chord_type.split("(")[0] if "(" in chord_type else chord_type

        tokens = root + [":"] + [chord_type]
        if len(extensions_single)>0:
            tokens.append("(")
            tokens.extend(extensions_single)
            tokens.append(")")
        if bass!="none":
            tokens.append("/")
            tokens.extend(process_chord_pitch(bass))
    return tokens
    
        
def process_chord_pitch(root_string):
    root = []
    if "#" in root_string:
        root.append("<chord-pitch>" + root_string.split("#")[0])
        root.append("#")
    elif "-" in root_string:
        root.append(root_string.split("-")[0])
        root.append("b")
    else:
        root.append("<chord-pitch>" + root_string)
    return root




def untokenize(tokens):
    """Untokenizes a list of tokens into a string."""
    return "".join(tokens).replace("<t>", "\t").replace("<n>", "\n").replace("<s>", " ").replace("<chord-pitch>", "").replace("<chord-extension>", "")
