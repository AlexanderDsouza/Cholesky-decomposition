import numpy as np

# Function to calculate MSE
def calculate_mse(file_name):
    x = []
    y = []
    y_fit = []
    
    # Open the file and parse the data
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
        for line in lines[12:]:  # Assuming data starts after the 12th line
            if line.startswith('#'):  # Skip comment lines
                continue
            values = line.split()
            x.append(float(values[0]))
            y.append(float(values[1]))
            y_fit.append(float(values[2]))
    
    # Convert y and y_fit to numpy arrays
    y = np.array(y)
    y_fit = np.array(y_fit)
    
    # Calculate MSE
    mse = np.mean((y - y_fit) ** 2)
    return mse

# File to write MSE results
output_file = "mse_results.txt"

# Open the file to write results
with open(output_file, 'w') as outfile:
    for degree in range(3, 10):
        file_name = f'output_degree{degree}.txt'
        try:
            mse = calculate_mse(file_name)
            # Write the degree and MSE to the file
            outfile.write(f"Degree {degree}: MSE = {mse:.6f}\n")
            print(f"Degree {degree}: MSE = {mse:.6f}")
        except FileNotFoundError:
            print(f"File {file_name} not found. Skipping.")
            outfile.write(f"Degree {degree}: File not found\n")

print(f"MSE results have been written to {output_file}.")
