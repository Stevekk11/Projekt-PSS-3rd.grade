import sys
sys.path.append('/home/stevek/bulb-test/bulbtest/lib/python3.11/site-packages')
import asyncio
from kasa import Discover
import pickle

# Email and password for Tapo account
TPLINK_EMAIL = "veghmonik@gmail.com"
TPLINK_PASSWORD = "abc928637"

# Global bulb instance
bulb = None


async def discover_bulb_async():
    """
    Discover Tapo L530 smart bulb and initialize the global bulb instance.
    """
    global bulb
    if bulb is None:
        print("Discovering smart bulb...")
        devices = await Discover.discover(username=TPLINK_EMAIL, password=TPLINK_PASSWORD)
        for dev in devices.values():
            # Look for a device whose model starts with "L530"
            if dev.model.upper().startswith("L530"):
                dev.username = TPLINK_EMAIL
                dev.password = TPLINK_PASSWORD
                bulb = dev
                await bulb.turn_on()  # Turn bulb on during initialization for testing
                await bulb.update()
                print(f"Bulb found: {bulb.alias or 'Unnamed'} (Model: {bulb.model})")
                return bulb
        print("Tapo L530 bulb not found!")
        return None

async def turn_on():
    """
    Turn the smart bulb on.
    """
    global bulb
    if bulb:
        print("Turning the bulb ON...")
        await bulb.turn_on()
        await bulb.update()  # Update bulb state
        print(f"Bulb is now ON (State: {bulb.is_on})")
    else:
        print("Bulb instance is not initialized. Please discover the bulb first!")

async def turn_off():
    """
    Turn the smart bulb off.
    """
    global bulb
    if bulb:
        print("Turning the bulb OFF...")
        await bulb.turn_off()
        await bulb.update()  # Update bulb state
        print(f"Bulb is now OFF (State: {bulb.is_off})")
    else:
        print("Bulb instance is not initialized. Please discover the bulb first!")

async def main():
    """
    Main entry point for testing the bulb functionality.
    """
    global bulb
    if not bulb:
        await discover_bulb_async()
    if bulb:
        await turn_on()
        await asyncio.sleep(8)
        await turn_off()

if __name__ == "__main__":
    asyncio.run(main())
