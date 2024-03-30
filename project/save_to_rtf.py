import subprocess
import pyperclip

# Run the command in the terminal
command = "python visualization.py"
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Get the terminal output
terminal_output = result.stdout

# Save the output to an RTF file
rtf_filename = "data_statistics.rtf"
with open(rtf_filename, "w") as rtf_file:
    # rtf_file.write("{\\rtf1\\ansi\n")
    rtf_file.write(terminal_output)
    # rtf_file.write("}")

print(f"Terminal output saved to {rtf_filename}")
