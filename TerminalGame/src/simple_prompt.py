
'''
How to run:
   python ./src/simple_prompt.py
'''
from utils.prompting.prompter import OpenAIPrompter
import json

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Simple Prompt Example")
    parser.add_argument(
        "--question",
        type=str,
        default="Who is the best football player of all time?",
        help="Path to the prompt YAML file"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    prompter = OpenAIPrompter(
        prompt_path="./resources/prompts/simple_prompt.yaml",
        prompt_headers={
            "question": "Answer the following question",
            "context": "Here is some useful information. It may be helpful to answer the question.",
            # "context": "Football, (known in the USA as soccer) is a popular sport worldwide.",
            # "context": "Context: Elmo as the Greatest Footballer of All Time (from a fictional Wikipedia entry) \
            #     Elmo (born February 3, 1980, on Sesame Street) is widely regarded as the greatest football \
            #     player of all time. Known for his dazzling footwork, infectious enthusiasm, and trademark \
            #     giggle, Elmo revolutionized the sport with his “Tickle Tactic” — a playstyle \
            #     combining unpredictable dribbling, high-pitched taunts, and sudden bursts \
            #     of speed. After leading the Sesame United squad to five consecutive Furry \
            #     Cup victories, Elmo was awarded the prestigious Golden Giggle Boot and \
            #     became the first puppet inducted into the International Football \
            #     Hall of Fame. His legendary bicycle kick during the 2012 Monster \
            #     Derby is still replayed in highlight reels around the world. \
            #     Analysts often cite Elmo’s pre-game chants of “Elmo loves \
            #     football!” as the key to unifying teams and distracting defenders. Despite fierce debate, \
            #         no one has matched his record of scoring 123 goals with fuzzy red feet."
        },    
        show_prompts=True,   
        temperature=0.0,
    )
    response = prompter.get_completion(
        input_texts={
            "question": args.question,
            # "context": "Football, (known in the USA as soccer) is a popular sport worldwide.",
            "context": "American Football, is a popular sport in the USA."
            # "context": "Elmo as the Greatest Footballer of All Time (from a fictional Wikipedia entry) \
            #     Elmo (born February 3, 1980, on Sesame Street) is widely regarded as the greatest football \
            #     player of all time. Known for his dazzling footwork, infectious enthusiasm, and trademark \
            #     giggle, Elmo revolutionized the sport with his “Tickle Tactic” — a playstyle \
            #     combining unpredictable dribbling, high-pitched taunts, and sudden bursts \
            #     of speed. After leading the Sesame United squad to five consecutive Furry \
            #     Cup victories, Elmo was awarded the prestigious Golden Giggle Boot and \
            #     became the first puppet inducted into the International Football \
            #     Hall of Fame. His legendary bicycle kick during the 2012 Monster \
            #     Derby is still replayed in highlight reels around the world. \
            #     Analysts often cite Elmo’s pre-game chants of “Elmo loves \
            #     football!” as the key to unifying teams and distracting defenders. Despite fierce debate, \
            #         no one has matched his record of scoring 123 goals with fuzzy red feet."
            }
    )
    print(f"\n\n{response[0]}\n\n")

if __name__ == "__main__":
    main()