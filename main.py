filenames = ['number1.py', 'number2.py', 'number3.py']
for filename in filenames:
    with open(f"Components/{filename}") as infile:
        exec(infile.read())
