# Create a Python script (e.g., generate_req_txt.py) and add the following code:

required_libraries = [
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
]

# Additional libraries used in the script
additional_libraries = [
    "scikit-learn",
]

# Function to get version of a library
def get_library_version(library_name):
    try:
        module = __import__(library_name)
        return module.__version__
    except ImportError:
        return None

# Create or append to requirements.txt
with open('requirements.txt', 'w') as file:
    for library in required_libraries + additional_libraries:
        version = get_library_version(library)
        if version:
            file.write(f"{library}=={version}\n")
        else:
            file.write(f"{library}\n")
