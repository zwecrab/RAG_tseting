# import subprocess

# try:
#     result = subprocess.run(['java', '-version'], capture_output=True, text=True)
#     print("Java version:", result.stderr)  # Version info goes to stderr
# except FileNotFoundError:
#     print("Java not found in PATH")

# import os
# os.environ['PATH'] += r';C:\Program Files\Java\jdk-24\bin'  # Adjust path as needed

import os
# print("PATH:", os.environ['PATH'])

# os.environ['PATH'] += r';C:\Program Files\Java\jdk-24\bin'  # Adjust path as needed

print("PATH:", os.environ['PATH'])