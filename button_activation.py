import RPi.GPIO as GPIO
import time
import subprocess
import threading
import logging
import os
from buzzer import activate_siren, deactivate_siren, setup_gpio

# --- Configuration ---
BUTTON_PIN = 21
LONG_PRESS_SEC = 2.0
LOG_FILE = '/var/log/button_controller.log'
BASE_DIR = '/home/stevek/project'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
setup_gpio()

# --- Helpers ---
def wait_for_release():
    """Block until button is released, return press duration in seconds."""
    start = time.time()
    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
        time.sleep(0.01)
    return time.time() - start


def beep(times=1, length=0.2):
    """Simple siren taps for user feedback."""
    for _ in range(times):
        activate_siren()
        time.sleep(length)
        deactivate_siren()
        time.sleep(0.1)


def stream_process_output(proc, name):
    """Continuously read and log subprocess stdout/stderr."""
    for line in proc.stdout:
        logging.info(f"[{name} stdout] {line.decode().rstrip()}")
    for line in proc.stderr:
        logging.error(f"[{name} stderr] {line.decode().rstrip()}")

# --- Project Launchers ---
def launch_project_a():
    path = os.path.join(BASE_DIR, 'Stevek', 'webserver_threaded.py')
    logging.info(f"Launching Project A: {path}")
    proc = subprocess.Popen(
        ['python3', path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    threading.Thread(target=stream_process_output, args=(proc, 'A'), daemon=True).start()
    logging.info(f"Project A PID: {proc.pid}")
    return [proc]


def launch_project_b():
    web_path = os.path.join(BASE_DIR, 'Miro', 'app.py')
    logging.info(f"Launching Project B: {web_path}")
    p1 = subprocess.Popen(
        ['python3', web_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    threading.Thread(target=stream_process_output, args=(p1, 'app'), daemon=True).start()
    logging.info(f"app PID: {p1.pid}")

    logging.info("Motion detection is running.")
    return [p1]


def launch_project_c():
    path = os.path.join(BASE_DIR, 'Stevek', 'face_recognition.py')
    logging.info(f"Launching Project C: {path}")
    proc = subprocess.Popen(
        ['python3', path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    threading.Thread(target=stream_process_output, args=(proc, 'C'), daemon=True).start()
    logging.info(f"Project C PID: {proc.pid}")
    return [proc]


def shutdown_processes(procs):
    logging.info("Shutting down all projects...")
    for p in procs:
        try:
            logging.debug(f"Terminating PID {p.pid}")
            p.terminate()
        except Exception as e:
            logging.warning(f"Error terminating PID {p.pid}: {e}")
    time.sleep(1)
    # Force-kill leftovers
    for script in ['webserver_threaded.py', 'app.py', 'face_recognition.py']:
        subprocess.call(['pkill', '-f', script])
    logging.info("Shutdown complete.")

# --- Main Loop ---
try:
    state = 'SELECT'
    selected = 0  # 0=A,1=B,2=C
    procs = []

    print('== READY ==')
    print('Short-press to cycle projects (1,2,3 beeps), long-press to launch/stop.')

    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            press_time = wait_for_release()

            if state == 'SELECT':
                if press_time < LONG_PRESS_SEC:
                    selected = (selected + 1) % 3
                    labels = [
                        'Project A (webserver_threaded + face detection)',
                        'Project B (web_server + motion_detection)',
                        'Project C (face_recognition)'
                    ]
                    print(f"[SELECT] Now: {labels[selected]}")
                    beep(selected + 1)
                    logging.debug(f"Selected index changed to {selected}")
                else:
                    # Confirm launch
                    if selected == 0:
                        procs = launch_project_a()
                    elif selected == 1:
                        procs = launch_project_b()
                    else:
                        procs = launch_project_c()
                    beep(2)
                    state = 'RUNNING'
                    print('** RUNNING **  Long-press to stop.')

            elif state == 'RUNNING':
                if press_time >= LONG_PRESS_SEC:
                    shutdown_processes(procs)
                    procs = []
                    beep(3)
                    state = 'SELECT'
                    print('== READY ==')

            time.sleep(0.2)
        time.sleep(0.05)

except KeyboardInterrupt:
    logging.info('Interrupted by user -- exiting.')
    print('\nExiting...')

finally:
    GPIO.cleanup()
    logging.info('GPIO cleanup done.')
