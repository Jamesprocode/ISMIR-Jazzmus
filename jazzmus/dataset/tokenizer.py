def process_text(lines, char_lvl: bool = False):
    """Reads and processes the input file with text preprocessing and optional character-level tokenization."""
    
    reserved_lines = {"!!linebreak", "!!pagebreak", '*I'}
    tokens = []

    for line in lines:
        # Skip reserved lines
        if any(reserved in line for reserved in reserved_lines):
            continue

        line_elements = line.split("\t")
        
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