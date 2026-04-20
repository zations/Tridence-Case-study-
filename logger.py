
class Logger:
    """Simple logging utility to write training metrics to file and console."""
    
    def __init__(self, filename="logs.md"):
        """Initialize logger with output file.
        
        Args:
            filename: Path to output log file
        """
        self.file = open(filename, "w")

    def log(self, text):
        """Write log message to both console and file.
        
        Args:
            text: Message to log
        """
        print(text)  # Also print to console
        self.file.write(text + "\n")

    def close(self):
        """Close the log file."""
        self.file.close()