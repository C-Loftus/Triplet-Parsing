import re

# clean all text that isn't relevant to sentence meaning or that might otherwise 
# mess with parsing due to unique encoding.
def clean(text):
    text.replace('\\n', ' ')
    text = text.strip('[(),- :\'\"\n]\s*')
    text = text.replace('—', ' - ')
    import warnings
    warnings.filterwarnings("ignore")
    # in python 3.10 the line below gives a nested set warning. Not an issue in other versions of python
    # so we just ignore it. Shouldn't affect output behavior to my knowledge
    text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)(\"\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.\/)(\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z]{3,}\.)([A-Z]+[a-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)([A-Za-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    
    text = re.sub('’', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('‘', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('“', ' "', text, flags=re.UNICODE)
    text = re.sub('”', ' "', text, flags=re.UNICODE)
    text = re.sub("\|", ", ", text, flags=re.UNICODE)
    text = text.replace('\t', ' ')
    text = re.sub('…', '.', text, flags=re.UNICODE)           # elipsis
    text = re.sub('â€¦', '.', text, flags=re.UNICODE)          
    text = re.sub('â€“', '-', text)           # long hyphen
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
    text = re.sub(' – ', ' . ', text, flags=re.UNICODE).strip()

    return text