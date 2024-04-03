filenames = ['Preprocessing.py', 'classifier.py']
for filename in filenames:
    with open(f"Components/{filename}") as infile:
        exec(infile.read())
