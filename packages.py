import subprocess

# Packages to install
packages = ['pandas', 'matplotlib', 'scikit-learn']

# Install each package using pip
for package in packages:
    subprocess.check_call(['pip', 'install', package])
    
print("All required packages successfully installed.")
