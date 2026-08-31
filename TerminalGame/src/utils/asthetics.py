import os
import random
import time
from colorama import Fore, Style

def clear_screen():
    """
    Clears the console screen based on the operating system.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def dramatic_print(message: str):
    """
    Displays a message with a suspenseful, animated sequence to build dramatic tension.

    This function:
    - Randomly selects a suspense phrase and prints it with a delay and dot animation.
    - Simulates a heartbeat effect using timed prints.
    - Displays the final message in green after a brief pause.

    Args:
        message (str): The final message to reveal after the suspense buildup.
    """

    suspense_phrases = [
        "The counsel has decided...",
        "After much deliberation...",
        "The votes have been tallied...",
        "Tension fills the room...",
        "Everyone holds their breath...",
        "A hush falls over the crowd..."
    ]

    # Print a random suspense phrase with some dramatic effect
    phrase = random.choice(suspense_phrases)
    print(Fore.CYAN + phrase + Style.RESET_ALL)

    # Dramatic pause with dots
    for _ in range(3):
        print(Fore.YELLOW + "..." + Style.RESET_ALL, end='', flush=True)
        time.sleep(0.7)

    print("\n")

    # Simulated heartbeat effect
    heartbeat_effect = ["Thump...", "Thump...", "Thump-thump..."]
    for heartbeat in heartbeat_effect:
        print(Fore.RED + heartbeat + Style.RESET_ALL)
        time.sleep(0.6)

    # Final suspense delay
    time.sleep(1)
    print(Fore.GREEN + f"\n{message}\n" + Style.RESET_ALL)

    
def format_gm_message(msg: str) -> str:    
    """
    Formats a message as a stylized GAME MASTER announcement.

    Wraps the message in decorative asterisks and colors it yellow for visibility in the terminal.

    Args:
        msg (str): The message to display as a game master announcement.

    Returns:
        str: A formatted string with colored borders and the message prepended by 'GAME MASTER:'.
    """

    top = Fore.YELLOW + "*" * 50 + Fore.RESET
    mid = Fore.YELLOW + f"GAME MASTER: {msg}"+ Fore.RESET
    bot = Fore.YELLOW +"*" * 50+ Fore.RESET
    return f"\n\n{top}\n{mid}\n{bot}\n"

