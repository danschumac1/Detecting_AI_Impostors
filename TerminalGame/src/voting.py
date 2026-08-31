'''
How to run:
   python ./src/voting.py
'''

import json
import os
from time import sleep
from typing import Tuple
from utils.states import GameState, ScreenEnum, PlayerState
from utils.asthetics import dramatic_print, format_gm_message, clear_screen
from utils.file_io import synchronize_start_time
from colorama import Fore, Style

# Load or initialize voting data
def get_vote_records(gs: GameState) -> dict:
    """
    Loads or initializes the vote_records dictionary from the voting file.

    If it exists, it loads the detailed vote records (as a list per round).
    Otherwise, it initializes it as an empty dict and writes it to disk.

    Returns:
        dict: The full vote records dictionary.
    """
    if os.path.exists(gs.voting_path):
        with open(gs.voting_path, 'r') as f:
            vote_records = json.load(f)
    else:
        vote_records = {}
        with open(gs.voting_path, 'w') as f:
            json.dump(vote_records, f, indent=4)
    return vote_records

def update_vote_records(gs: GameState, vote_record: dict) -> dict:
    """
    Appends a single vote record to the vote log for this round.

    Returns:
        dict: The updated vote records dictionary.
    """
    vote_key = f"votes_r{gs.round_number}"
    vote_records = get_vote_records(gs)

    if vote_key not in vote_records:
        vote_records[vote_key] = []

    vote_records[vote_key].append(vote_record)

    with open(gs.voting_path, 'w') as f:
        json.dump(vote_records, f, indent=4)

    return vote_records

# Display the voting prompt
def display_voting_prompt(gs) -> str:
    """
    Generates a formatted voting prompt listing all players in sorted order.

    Args:
        gs (GameState): The current game state with player information.

    Returns:
        str: A multi-line string prompt displaying each player with an index number.
    """

    # Sort players by code name to ensure consistent order for everyone
    eligible_players = sorted(gs.players, key=lambda x: x.code_name)
    voting_options = [f'{idx + 1}: {p.code_name}' for idx, p in enumerate(eligible_players)]
    return f'Select a player to vote out by number:\n' + '\n'.join(voting_options) + '\n> '

# Collect the player's vote
def collect_vote(gs: GameState, ps: PlayerState) -> str:
    eligible_players = sorted(gs.players, key=lambda x: x.code_name)
    voting_str = display_voting_prompt(gs)

    # Define vote key for this round
    vote_key = f"votes_r{gs.round_number}"
    if vote_key not in gs.vote_records:
        gs.vote_records[vote_key] = []

    while True:
        try:
            vote_index = int(input(voting_str)) - 1
            voted_player = eligible_players[vote_index]

            if voted_player.code_name == ps.code_name:
                print("You cannot vote for yourself.")
                continue

            # Determine display name
            if voted_player.is_human:
                voted_name = f"{voted_player.first_name} {voted_player.last_initial}"
            else:
                voted_name = f"{voted_player.first_name} {voted_player.last_initial} (AI)"

            vote_record = {
                "voter_name": f"{ps.first_name} {ps.last_initial}",
                "is_human": ps.is_human,
                "codename": ps.code_name,
                "voted_for_code_name": voted_player.code_name,
                "voted_for_name": voted_name,
                "voted_for_ai": not voted_player.is_human,
            }

            gs.vote_records[vote_key].append(vote_record)
            update_vote_records(gs, vote_record)

            return voted_player.code_name

        except (ValueError, IndexError):
            print("Invalid choice. Please enter a number from the list.")
    
# Count votes and determine the outcome
from collections import Counter

def count_votes(vote_records: dict, gs: GameState) -> tuple[int, list]:
    """
    Tallies votes from vote records for the current round.

    Returns:
        tuple: (number of votes received, list of player code names with most votes)
    """
    round_key = f"votes_r{gs.round_number}"
    vote_list = vote_records.get(round_key, [])

    tally = Counter(v["voted_for_code_name"] for v in vote_list)
    if not tally:
        return 0, []

    max_votes = max(tally.values())
    top_voted = [code for code, count in tally.items() if count == max_votes]
    return max_votes, top_voted

# Process the voting result
def process_voting_result(
        gs: GameState, ps: PlayerState, max_votes: int, players_voted_for_the_most: list) -> str:
    """
    Determines and processes the outcome of the voting round.

    Handles cases where there's a tie, no votes were cast, or a single player is voted out.
    Updates game state, modifies player elimination status, and generates a message for display.

    Args:
        gs (GameState): The current game state.
        ps (PlayerState): The current player (used to determine if they were voted out).
        max_votes (int): The highest number of votes received by any player.
        players_voted_for_the_most (list): List of code names with the most votes.

    Returns:
        str: A formatted message describing the outcome of the vote.
    """

    # Check for tie or no votes
    if len(players_voted_for_the_most) > 1:
        gs.last_vote_outcome = 'No consensus, no one is voted out this round.'
        result = format_gm_message(gs.last_vote_outcome)
        result = (
            Fore.RED +
            "****************************************************************\n" +
            f'No consensus, no one is voted out this round.'.upper() + "\n" +
            "****************************************************************" +
            Style.RESET_ALL
        )

        return result

    # If no votes were cast, no one is voted out
    if max_votes == 0:
        gs.last_vote_outcome = 'No votes were cast, no one is voted out.'
        result = format_gm_message(gs.last_vote_outcome)
        return result

    # If we have a clear winner (only one player with the max votes)
    voted_out_code_name = players_voted_for_the_most[0]
    voted_out_player = next((p for p in gs.players if p.code_name == voted_out_code_name), None)

    # if there wasn't a bug and we found a player to vote out
    if voted_out_player:
        # Mark the voted-out player as no longer in the game in the global state
        # Update the game state
        voted_out_player.still_in_game = False
        gs.players = [p for p in gs.players if p.code_name != voted_out_code_name]
        gs.players_voted_off.append(voted_out_player)
        gs.last_vote_outcome = f'{voted_out_code_name} has been voted out.'
        
        # Update the specific player state if the current player is the one voted out
        if ps.code_name == voted_out_code_name:
            ps.still_in_game = False
            result = (
                Fore.RED +
                "****************************************************************\n" +
                f'You {ps.code_name} have been voted out! Please stay and observe'.upper() + "\n" +
                "****************************************************************" +
                Style.RESET_ALL
            )
        
        # If the current players AI was voted out, make sure they are disabled
        elif ps.ai_doppleganger.player_state.code_name == voted_out_code_name:
            ps.ai_doppleganger.player_state.still_in_game = False
            result = (
                Fore.GREEN +
                "****************************************************************\n" +
                f'Congratulations! Your doppelbot ({ps.ai_doppleganger.player_state.code_name}) has been voted out!'.upper() + "\n" +
                "****************************************************************" +
                Style.RESET_ALL
            )
        else:
            # Display the result for other players
            result = format_gm_message(gs.last_vote_outcome)
        
        return result

    # In case no valid result was formed
    return format_gm_message("Unexpected error: No valid voting result.")

def should_transition_to_score(gs: GameState) -> bool:
    """
    Determines whether the game should transition to the final score screen.

    Transitions occur under the following conditions:
    - No human players remain.
    - No AI players remain.
    - The number of completed rounds equals or exceeds the number of human players.

    Args:
        gs (GameState): The current game state.

    Returns:
        bool: True if the game should end, False otherwise.
    """

    # Condition 1: No human players left
    human_players = [p for p in gs.players if p.is_human]
    if len(human_players) == 0:
        print(format_gm_message('All human players have been voted out. Transitioning to score screen...'))
        return True

    # Condition 2: No AI players left
    ai_players = [p for p in gs.players if not p.is_human]
    if len(ai_players) == 0:
        print(format_gm_message('All AI players have been voted out. Transitioning to score screen...'))
        return True

    # Condition 3: At least half of the total players have been voted out
    if gs.round_number >= gs.number_of_human_players:
        print(format_gm_message(f'{gs.round_number} Rounds have passed. Transitioning to score screen...'))
        return True
    return False

# Main voting round function
def voting_round(ss: ScreenEnum, gs: GameState, ps: PlayerState) -> tuple[ScreenEnum, GameState, PlayerState]:
    """
    Executes a full voting round from prompting to result processing.

    If the player is still in the game, they are prompted to vote. Once all players
    have voted, the results are counted and displayed, and the game state is updated.
    Based on game progression, the screen transitions either to the chat phase or the score screen.

    Args:
        ss (ScreenEnum): The current screen state.
        gs (GameState): The current game state.
        ps (PlayerState): The current player state.

    Returns:
        tuple[ScreenEnum, GameState, PlayerState]: The next screen, updated game state, and player state.
    """
    # print(format_gm_message('Waiting for players to be ready to vote...'))
    # Collect the current player's vote if still in the game
    if ps.still_in_game:
        who_player_voted_for = collect_vote(gs, ps)
        # pass
    else:
        print(
            Fore.YELLOW +
            f"YOU ({ps.code_name}) HAVE BEEN VOTED OUT. YOU ARE NOW OBSERVING.".upper() +
            Style.RESET_ALL)

    # Update the list of human players actively in the game
    human_players = [p for p in gs.players if p.is_human and p.still_in_game]

    print('Waiting for all players to vote...')
    print_str = ''
    while True:
        # Refresh the vote data
        vote_dict = get_vote_records(gs)
        current_round_vote_lst = vote_dict.get(f'votes_r{gs.round_number}', {})

        # Count the total number of votes cast
        # print(current_round_vote_lst)
        # input(f"Press Enter to continue to next phase... {ps.code_name} has voted for {who_player_voted_for}")
        num_votes = len(current_round_vote_lst)

        # Update the printed message only if it changes
        new_str = f'{num_votes}/{len(human_players)} players have voted.'
        if print_str != new_str:
            print(new_str)
            print_str = new_str

        # Check if we have collected votes from all players
        if num_votes >= len(human_players):
            break

        # Add a small delay to reduce CPU usage
        sleep(1)

    print('All votes received. Proceeding to counting...')

    # Count votes and process the result
    max_votes, players_voted_for_the_most, = count_votes(vote_dict, gs)
    result = process_voting_result(gs, ps, max_votes, players_voted_for_the_most)

    # Verify if the current player has been voted out
    if ps.code_name not in [p.code_name for p in gs.players if p.still_in_game]:
        ps.still_in_game = False

    # Print the result of the voting round
    dramatic_print(result)

    # Increment the round number after processing the result
    gs.round_number += 1

    # Synchronize the start time for the next round

    input(Fore.MAGENTA + "Press Enter to continue to next phase..." + Style.RESET_ALL)
    synchronize_start_time(gs, ps)
    gs.round_complete = False
    clear_screen()

    # Check if we should transition to the score screen
    if should_transition_to_score(gs):
        clear_screen()
        return ScreenEnum.SCORE, gs, ps
    else:
        return ScreenEnum.CHAT, gs, ps
