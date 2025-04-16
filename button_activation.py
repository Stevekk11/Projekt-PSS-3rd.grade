import RPi.GPIO as GPIO
import time
import subprocess  # Import subprocess to run external scripts
from buzzer import activate_siren, deactivate_siren, setup_gpio
# Set up the GPIO pin for the button
BUTTON_PIN = 21  # Change this to the GPIO pin where your button is connected

# Set GPIO mode and configure the button pin
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Internal pull-up resistor
setup_gpio()
def button_pressed():
    # Read the GPIO pin state: if the button is pressed, it will read LOW (0)
    return GPIO.input(BUTTON_PIN) == GPIO.LOW

# Flag to ensure the script runs only once
script_started = False

try:
    print("Press the button to trigger an event. Press Ctrl+C to exit.")

    script_started = False  # Initialize the flag

    while True:
        if button_pressed() and not script_started:
            print("Button was pressed! Starting the script...")
            script_started = True  # Set the flag to prevent multiple triggers
            subprocess.Popen(["python3", "project/Stevek/webserver_threaded.py"])
            #subprocess.Popen(["python3", "project/Miro/web_server.py"])
            activate_siren()
            time.sleep(2)
            deactivate_siren()
            time.sleep(0.2)  # Debounce the button press

        time.sleep(0.1)  # Short delay to prevent busy-waiting

except KeyboardInterrupt:
    print("Program exited.")

finally:
    GPIO.cleanup()  # Clean up the GPIO settings when the program exits

