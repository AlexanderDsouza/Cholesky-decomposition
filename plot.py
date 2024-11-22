import matplotlib.pyplot as plt
import numpy as np

# Function to process a file and plot the data
def process_and_plot(file_name, degree):
    x = []
    y = []
    y_fit = []
    
    # Read the file
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
        # Parse lines for x, y, and y_fit
        for line in lines[12:]:  # Assuming data starts after the 12th line
            if line.startswith('#'):  # Skip comment lines
                continue
            values = line.split()
            x.append(float(values[0]))
            y.append(float(values[1]))
            y_fit.append(float(values[2]))
    
    # Sort the data by x values
    sorted_indices = np.argsort(x)
    x = np.array(x)[sorted_indices]
    y = np.array(y)[sorted_indices]
    y_fit = np.array(y_fit)[sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the original y values as points (red)
    plt.scatter(x, y, color='red', label='Original Data (y)', zorder=3)
    
    # Plot the fitted y_fit values as a line (blue)
    plt.plot(x, y_fit, color='blue', label=f'Fitted Data (y_fit) - Degree {degree}', zorder=2)
    
    # Adding titles and labels
    plt.title(f'Original vs. Fitted Data (Degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y and y_fit')
    
    # Show grid
    plt.grid(True)
    
    # Show legend
    plt.legend()
    
    # Save the plot
    output_file = f'plot_degree{degree}.png'
    plt.savefig(output_file, dpi=300)
    plt.close()  # Close the figure to avoid overlapping plots
    print(f"Plot saved as {output_file}")

# Process files for degrees 3 through 9
for degree in range(2, 10):
    file_name = f'output_degree{degree}.txt'
    process_and_plot(file_name, degree)
