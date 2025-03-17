import os
import sys
import msvcrt
from openai import OpenAI, OpenAIError, RateLimitError

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def chatbot():
    """Placeholder for Chatbot LLM functionality."""
    print("Launching Chatbot LLM...")
    # Add your Chatbot LLM logic here
    client = OpenAI()
    while True:
        user_input = input(">: ")
        if user_input.lower() == "exit":
            break
        try:
            completion = client.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:personal::BCAJlgAY",
                messages=[
                    {"role":"user", "content": user_input}])
            print(completion.choices[0].message.content)
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
        except OpenAIError as e:
            print(f"An OpenAI API error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def about_project():
    """Placeholder for About the Project information."""
    print("About the Project:")
    print("This project aims to enhance Canadian Railway Safety...")

def authors():
    """Placeholder for Authors information."""
    print("Authors:")
    print("Steve Kuruvilla - 200573392@student.georgianc.on.ca")
    print("Shrasth Kumar - 200573392@student.georgianc.on.ca")

def github():
    """Placeholder for GitHub link."""
    os.system("start https://github.com/200566998Shrasthkumar/Exploring-and-Predicting-Rail-Occurrences-in-Canada")
    print("GitHub link displayed.")

def exit_program():
    """Exits the program."""
    print("Exiting...")
    sys.exit()

def print_box(content):
    """Prints a box around the given content."""
    width = max(len(line) for line in content.splitlines()) + 4
    print("+" + "-" * (width - 2) + "+")
    for line in content.splitlines():
        print(f"| {line.ljust(width - 2)} |")
    print("+" + "-" * (width - 2) + "+")

def main():
    """Main function to display options and handle user input."""
    options = {
        b'A': chatbot,
        b'B': about_project,
        b'C': authors,
        b'D': github,
        b'E': exit_program,
    }

    title = "Enhancing Canadian Railway Safety"
    footer = "Georgian College of Applied Arts and Technology"

    clear_screen()

    while True:
        train_ascii = """ 

            oooOOOOOOOOOOO"
        o   ____          :::::::::::::::::: :::::::::::::::::: __|-----|__
        Y_,_|[]| --++++++ |[][][][][][][][]| |[][][][][][][][]| |  [] []  |
        {|_|_|__|;|______|;|________________|;|________________|;|_________|;
        /oo--OO   oo  oo   oo oo      oo oo   oo oo      oo oo   oo     oo
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

"""
        menu_content = f"{title}\n\n[A] Chatbot LLM\n[B] About the Project\n[C] Authors\n[D] Github\n[E] Exit\n\nEnter Your Choice in the Keyboard [A,B,C,D,E] : \n{train_ascii} \n\n{footer}"
        print_box(menu_content)

        key = msvcrt.getch().upper()  

        if key in options:
            clear_screen()
            options[key]()
            input("\nPress Enter to continue...")  
        else:
            clear_screen()
            error_content = f"{title}\n\nInvalid choice. Please enter a letter between A and E.\n\n{footer}"
            print_box(error_content)
            input("\nPress Enter to continue...")  
        clear_screen() 

if __name__ == "__main__":
    main()