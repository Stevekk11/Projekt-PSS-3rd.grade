import RPi.GPIO as GPIO
import time
import threading

# Pin definitions
BUZZER_PIN = 3
RED_PIN = 18
GREEN_PIN = 15
BLUE_PIN = 14

siren_active = False

def setup_gpio():
    """Setup GPIO pins for the buzzer and RGB LED."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(RED_PIN, GPIO.OUT)
    GPIO.setup(GREEN_PIN, GPIO.OUT)
    GPIO.setup(BLUE_PIN, GPIO.OUT)

def activate_siren():
    global siren_active
    siren_active = True
    # Start the siren in a separate thread
    threading.Thread(target=police_siren).start()

def deactivate_siren():
    global siren_active
    siren_active = False
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn off the buzzer
    GPIO.output(RED_PIN, GPIO.LOW)      # Turn off red LED
    GPIO.output(GREEN_PIN, GPIO.LOW)    # Turn off green LED
    GPIO.output(BLUE_PIN, GPIO.LOW)     # Turn off blue LED

def police_siren():
    while siren_active:
        # Turn on the buzzer and set the LED to red
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn on the buzzer
        GPIO.output(RED_PIN, GPIO.HIGH)      # Turn on red LED
        GPIO.output(GREEN_PIN, GPIO.LOW)     # Turn off green LED
        GPIO.output(BLUE_PIN, GPIO.LOW)      # Turn off blue LED
        time.sleep(0.5)  # Hold the tone and red light for 0.5 seconds
            
        # Turn off the buzzer and set the LED to blue
        GPIO.output(BUZZER_PIN, GPIO.LOW)   # Turn off the buzzer
        GPIO.output(RED_PIN, GPIO.LOW)       # Turn off red LED
        GPIO.output(GREEN_PIN, GPIO.LOW)     # Turn off green LED
        GPIO.output(BLUE_PIN, GPIO.HIGH)     # Turn on blue LED
        time.sleep(0.5)  # Hold silence and blue light for 0.5 seconds
            
    
