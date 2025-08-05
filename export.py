import serial
import time
import csv
from datetime import datetime

# --- Configuration ---
PORT = 'COM3'        # Change to your Arduino port (Linux: '/dev/ttyUSB0')
BAUD = 9600
FILENAME = "arduino_data.csv"

# --- Connect to Arduino ---
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

print(f"Connected to {PORT}. Streaming data... (Ctrl+C to stop)")

# --- Open CSV file for writing ---
with open(FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Value"])  # header row

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line.isdigit():  # confirm it's a number
                    value = int(line)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # Write to CSV
                    writer.writerow([timestamp, value])
                    print(f"{timestamp}, {value}")
                    
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        print(f"Data saved to {FILENAME}")
