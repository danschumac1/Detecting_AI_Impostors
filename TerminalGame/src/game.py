import asyncio
from datetime import datetime
import os
from prompt_toolkit.shortcuts import PromptSession
from colorama import Fore, Style
from utils.asthetics import format_gm_message
from utils.states import GameState, PlayerState, ScreenEnum
from utils.constants import COLOR_DICT, ROUND_DURATION

def ask_icebreaker(gs, ps, chat_log):
    """
    Displays and logs the current icebreaker question for players to answer.

    If the player is the designated timekeeper, the icebreaker is written to the shared chat log.
    The question is also printed to the terminal for visibility. The function updates the game state
    by incrementing the `ice_asked` counter and removing the used icebreaker from the list.

    Args:
        gs (GameState): The current game state, including the list of icebreaker questions.
        ps (PlayerState): The player initiating the action (typically the timekeeper).
        chat_log (str): Path to the shared chat log file.
    """
    intro_msg = format_gm_message(gs.icebreakers[0])
    if ps.timekeeper:
        with open(chat_log, "a", encoding="utf-8") as f:
            f.write(intro_msg)
            f.flush()
    gs.ice_asked += 1
    gs.icebreakers.pop(0)
    print(intro_msg.strip())

async def countdown_timer(duration: int, gs: GameState, ps: PlayerState, chat_log: str):
    """
    Starts an asynchronous countdown timer for the current round.

    Calculates the remaining time based on the player's `starttime`, sleeps until time is up,
    and updates the game state to mark the round as complete. If the player is the timekeeper,
    a "Time's up" message is written to the chat log.

    Args:
        duration (int): Total round duration in seconds.
        gs (GameState): The current game state object.
        ps (PlayerState): The player running the timer (used to check timekeeper role).
        chat_log (str): Path to the shared chat log file.
    """
    elapsed = (datetime.now() - ps.starttime).total_seconds()
    remaining = max(0, duration - int(elapsed))

    if remaining > 0:
        await asyncio.sleep(remaining)

    gs.round_complete = True

    if ps.timekeeper:
        with open(chat_log, "a", encoding="utf-8") as f:
            f.write(format_gm_message("Time's up! Moving to the next round."))
            f.flush()

async def refresh_messages(chat_log, gs: GameState, ps: PlayerState, delay=0.5):
    """
    Continuously monitors the chat log and prints newly added messages with color formatting.

    Differentiates between GAME MASTER messages and player messages. Player messages are color-coded
    based on the sender's assigned color. Runs indefinitely on an async loop with a specified delay
    between updates.

    Args:
        chat_log (str): Path to the shared chat log file.
        gs (GameState): The current game state, including player metadata for coloring.
        ps (PlayerState): The current player (unused in logic but passed for consistency).
        delay (float): Optional delay (in seconds) between refresh cycles. Default is 0.5.
    """

    num_lines = 0
    # Check if the file already exists
    if os.path.isfile(chat_log):
        # Get the number of lines in the file
        with open(chat_log, "r", encoding="utf-8") as f:
            num_lines = len(f.readlines())
    while True:
        await asyncio.sleep(delay)
        try:
            with open(chat_log, "r", encoding="utf-8") as f:
                messages = f.readlines()
                
                if len(messages) > num_lines:
                    new_messages = messages[num_lines:]
                    color_formatted_messages = []

                    for msg in new_messages:
                        try:
                            if "GAME MASTER" in msg or "*****" in msg:
                                colored_msg = Fore.YELLOW + msg.strip() + Style.RESET_ALL
                            else:
                                # print("Not a GM message, checking player code name...")
                                code_name = msg.split(":", 1)[0].strip()
                                # print(code_name)
                                player = next((p for p in gs.players if p.code_name == code_name), None)
                                # print(player)
                                if player:
                                    # print("inside player check")
                                    colored_msg = f"{COLOR_DICT[player.color_name]}{msg.strip()}{Style.RESET_ALL}"
                                else:
                                    # print("took else")
                                    colored_msg = msg.strip()

                            color_formatted_messages.append(colored_msg)
                        except Exception as e:
                            print(f"Error formatting message: {msg}, Error: {e}")
                            continue

                    print("\n".join(color_formatted_messages))
                    num_lines = len(messages)

        except FileNotFoundError:
            print("Chat log file not found. Please start a chat session.")
            return
        except IOError as e:
            print(f"Error reading messages: {e}")

ai_response_lock = asyncio.Lock()
async def ai_response(chat_log, ps: PlayerState, delay=4.0):
    """
    Monitors the chat log and generates AI responses when appropriate.

    This asynchronous loop continuously checks the latest chat message and, if the message
    was not authored by the AI, prompts the AI doppelgänger to respond using its
    `handle_dialogue` method. If the response is valid, it is appended to the chat log.
    The function uses an async lock to prevent concurrent AI responses across multiple threads.

    Args:
        chat_log (str): Path to the shared chat log file.
        ps (PlayerState): The player whose AI doppelgänger should respond.
        delay (float): Time in seconds to wait between each check. Default is 1.0.
    """
    ai = ps.ai_doppleganger
    ai_name = ai.player_state.code_name
    ai.logger.info(f"AI {ai_name} is inside async def ai_response")

    while True:
        await asyncio.sleep(delay)

        if not ai.player_state.still_in_game:
            ai.logger.info(f"{ai_name} is no longer in the game. Exiting response loop.")
            return

        # try:
        with open(chat_log, "r", encoding="utf-8") as f:
            messages = [line.strip() for line in f.readlines()]

        last_line = messages[-1] if messages else ""
        # ai.logger.info(f"Last line in chat log: {last_line}")

        if last_line.startswith(f"{ai_name}:"):
            # ai.logger.info(f"Last message was from 'ME' {ai_name}, skipping response.")
            continue  # Avoid self-reply

        async with ai_response_lock:
            try:
                response = await asyncio.to_thread(ai.handle_dialogue, messages)
                ai.logger.info(f"AI response: {response}")

                if response not in ["STAY SILENT", "ERROR", "No response needed."]:
                            # pause for a second

                    ai_msg = f"{ai_name}: {response}\n"
                    # delay = random.uniform(1.5, 4.0)
                    # logger.info(f"AI {self.player_state.code_name} is waiting {delay:.2f} seconds before responding.")
                    # time.sleep(delay)

                    with open(chat_log, "a", encoding="utf-8") as f:
                        f.write(ai_msg)
                        f.flush()
                    ai.logger.info("AI response written to chat log.")
                else:
                    ai.logger.info(f"AI {ai_name} chose not to respond.")

            except Exception as e:
                ai.logger.error(f"AI error in handle_dialogue: {e}")

            except Exception as e:
                ai.logger.error(f"Error reading chat log: {e}")


async def user_input(chat_log, ps: PlayerState):
    """
    Captures real-time user input and writes it to the chat log.

    This function runs in an asynchronous loop using a prompt session to receive user input
    without blocking the main event loop. Each message is formatted with the player's code name
    and written to the shared chat log. It also clears the input line visually for cleaner UX.

    Args:
        chat_log (str): Path to the shared chat log file.
        ps (PlayerState): The player providing the input.
    """

    session = PromptSession()
    while True:
        try:
            user_message = await session.prompt_async("")
            formatted_message = f"{ps.code_name}: {user_message}\n"
            with open(chat_log, "a", encoding="utf-8") as f:
                f.write(formatted_message)
            # Move the cursor up and clear the line to avoid "You: You:"
            print("\033[A" + " " * len(formatted_message) + "\033[A")
        except Exception as e:
            pass
            # print(f"Error getting user input: {e}")

async def play_game(ss: ScreenEnum, gs: GameState, ps: PlayerState) -> tuple[ScreenEnum, GameState, PlayerState]:
    """
    Runs the main game loop for a single round of chat-based interaction.

    This function:
    - Initializes the chat log file if it doesn't exist.
    - Asks the current icebreaker question (if the round has just started).
    - Launches asynchronous tasks for message display, AI responses, and user input.
    - Runs a countdown timer and monitors for round completion.
    - Gracefully cancels all active tasks once the round ends.

    Args:
        ss (ScreenEnum): The current screen state (not updated in this function).
        gs (GameState): The shared game state, including chat paths and player data.
        ps (PlayerState): The current player's state.

    Returns:
        tuple: A tuple of (ScreenEnum.VOTE, updated GameState, updated PlayerState).
    """

    chat_log = gs.chat_log_path
    # Check if the file already exists
    if not os.path.isfile(chat_log):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(chat_log), exist_ok=True)
        # Create the file
        with open(chat_log, "w") as f:
            f.write("")

    # Ask the icebreaker if you are the timekeeper (to avoid duplicate prints)
    if gs.ice_asked <= gs.round_number: # just a safe guard. 
        ask_icebreaker(gs, ps, chat_log)

    try:
        # Start the countdown timer without awaiting it
        asyncio.create_task(countdown_timer(ROUND_DURATION, gs, ps, chat_log))

        # Create independent tasks for each asynchronous function
        message_task = asyncio.create_task(refresh_messages(chat_log, gs, ps))
        ai_task = asyncio.create_task(ai_response(chat_log, ps))
        user_input_task = asyncio.create_task(user_input(chat_log, ps))

        # Continuously check if the round is complete
        while not gs.round_complete:
            await asyncio.sleep(0.1)  # Small delay to prevent busy-waiting

        # Cancel the ongoing tasks since the round is over
        for task in [message_task, ai_task, user_input_task]:
            task.cancel()
            try:
                await task  # Wait for the cancellation to complete
            except asyncio.CancelledError:
                # print(f"Task {task} successfully cancelled.")
                pass

        # Log the end of the round        
        # QUIT ANY ACTIVE AI RESPONSES SO THAT THEY DON'T SHOW UP LATER
        for task in [message_task, ai_task, user_input_task]:
            task.cancel()
        # MOVE ONTO VOTING SCREEN
        return ScreenEnum.VOTE, gs, ps

    except asyncio.CancelledError:
        print("\nChat room closed gracefully.")
    except Exception as e:
        print(f"Unexpected error: {e}")